# TriangleMix：面向 vLLM-Ascend 的单次 CANN Launch 稀疏 Prefill

## 1. 项目总览

本项目在昇腾 NPU 上实现 [TriangleMix](https://arxiv.org/abs/2507.21526)
论文提出的 Triangle 稀疏注意力，并以独立 wheel 插件接入
vLLM-Ascend。昇腾推理栈目前缺少能够真正跳过稀疏区域计算的
TriangleMix 算子；仅在 Python 层生成 mask，或先裁剪、重排 token，都不能
稳定兑现理论收益。因此本项目实现了直接读取 paged KV cache 的高性能
CANN 算子。

TriangleMix 利用长上下文中 decoding-time contribution sparsity：在选定的
Transformer 层，prefill 只保留开头的 attention sink 和当前 query 附近的
local window，跳过中间无贡献的 Q-K 区域。本实现的稀疏区域为
`[0, 8) ∪ [q-512, q+1)`，同时保留 prompt 前部的 dense causal prefix 和
最后 128 行的 dense causal tail。选定层的注意力复杂度由
`O(N²)` 降为 `O(N)`；未选层仍执行官方 dense attention。

自定义路径只用于 prefill。Decode、混合 decode/prefill batch、batch 大于
1、graph capture、不支持的 shape、版本和并行模式均自动回退官方
Fused Infer Attention（FIA）。

## 2. 插件安装

发布物是独立的 `vllm_ascend_trianglemix` wheel。它通过
`vllm.general_plugins` 的 `trianglemix` 入口自动注册，不包含顶层
`vllm/` 或 `vllm_ascend/` 文件，因此不会覆盖官方安装。

插件面向 vLLM-Ascend 0.23 的以下窄接口：

- `NPUModelRunner._build_attention_metadata`：附加 scheduler 持有的最终
  prompt 长度和 step token；
- `AscendAttentionMetadataBuilder.build`：为当前 step 缓存一次不可变路由
  metadata；
- `AscendAttentionBackendImpl.forward`：记录 layer name 并判断是否为选定层；
- `AscendAttentionBackendImpl.forward_fused_infer_attention`：符合条件的
  prefill 调用自定义单次-launch算子，其余调用原样转发官方 FIA。

wheel 同时携带私有 OPP tree 和 Torch adapter，只在功能启用后延迟加载。
安装、升级或卸载后必须重启全部 vLLM worker。

### 直接安装

```bash
python -m pip install \
  vllm_ascend_trianglemix-0.1.0-cp310-cp310-linux_aarch64.whl

export VLLM_PLUGINS=ascend,trianglemix
```

### 从源码构建

在下文指定的昇腾环境中构建。CANN 算子和 adapter 必须在 AArch64
目标机上生成。

```bash
# 1. 构建 CANN package
cd operator
bash build.sh
cd ..

# 2. 将算子安装到隔离 OPP staging root
export OPP_STAGE=/absolute/path/to/empty/opp-stage
mkdir -p "$OPP_STAGE"
operator/build_out/custom_opp_ubuntu_aarch64.run \
  --quiet \
  --install-path "$OPP_STAGE"

# 3. 准备 wheel 构建环境
export VLLM_ASCEND_SRC=/path/to/clean/vllm-ascend
python3.10 -m venv .venv-trianglemix
. .venv-trianglemix/bin/activate
python -m pip install "build==1.3.0" "pyproject-hooks==1.2.0"

# 4. 重建 adapter、组装 wheel 并执行 clean-install 审计
python package/tools/release_wheel_pipeline.py \
  --vllm-ascend-src "$VLLM_ASCEND_SRC" \
  --opp-root "$OPP_STAGE" \
  --output-dir /path/to/empty/release-dist \
  --report /path/to/new/release-wheel-report.json
```

流水线会重建并 strip adapter，检查 AArch64 ELF、私有路径泄漏和非法上游
覆盖，并在临时环境中执行安装/卸载审计。

## 3. CANN 算子说明

`TrianglePagedSparseAttention` 的固定计算几何为：

```text
Qwen3-8B；BF16；Hq=32；Hkv=8；D=128；GQA=4:1
KV page=128；query tile=32；outer KV tile=512；Cube inner tile=128
sink=8；local window=512；dense tail=128
```

一次 CANN MIX AIC/AIV launch 直接读取完整 query chunk、paged K/V cache 和
block table，并在内核内完成：

1. dense causal prefix；
2. sink/local-window sparse middle；
3. dense causal tail；
4. 跨全部 K/V 区间共享的 online softmax 与 output rescale；
5. 直接写入调用方持有的 BF16 output。

QK、PV 在 Cube 上计算；causal/local-window 边界 mask 和 online softmax
在 Vector 上处理。outer KV tile 为 512，内部按 128 列执行 Cube
QK/PV，并复用 CANN 9.0.1 FIA 的 paged block-table 地址映射。被稀疏策略
跳过的逻辑 token 不进入 QK 或 PV。该实现消除了 Python
侧 K/V gather、token pruning、pack、cache reorder、额外 output 分配，
也将“dense prefix → sparse middle → dense tail”三次 launch 融为一次。

关键问题及处理如下：

- paged KV 的逻辑页可能映射到非连续物理页：QK 与 PV 使用同一份
  block-table 映射，直接从 BSND cache 读取；
- 不同稀疏区间必须共享一个 softmax：所有 tile 持续维护 row max、row sum
  和 output accumulator，跨区间执行统一 rescale；
- CANN LibApi workspace 与算子用户区分别计量：显式申请 16 MiB
  LibApi 区和 1,839,104 字节用户区，共 18,616,320 字节；
- final drain 的 `EVENT_ID2` token 在初始化和回收处建立对称
  `MTE3_V` event 生命周期；
- `validEnd=127` 等边界会产生未按 32 字节对齐的 UB 地址：改为 aligned
  base 加显式 bit-mask `Duplicate`；
- Python 导入存在循环依赖：插件先导入 `vllm_ascend.ops`，再导入
  `vllm_ascend.attention.attention_v1`；
- 短序列中 launch 和路由开销可能超过节省的计算：使用离线 crossover
  阈值自动回退 FIA，并按 worker/rank 记录每请求 hit、fallback reason、
  routing、launch 和 enqueue 计数。

## 4. 测试说明

### 已验证环境

| 组件 | 版本或配置 |
| --- | --- |
| 操作系统 | Linux AArch64 |
| NPU | Ascend 910B3，单卡 |
| Python | 3.10 |
| CANN | 9.0.1 |
| vLLM | 0.23.0（允许 `+empty` 等本地后缀） |
| vLLM-Ascend | 0.23.0rc1 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0.post2 |
| 模型 | Qwen3-8B |

插件对上述完整指纹和固定算子几何执行 fail-closed 门禁。其他
pre/post/dev 版本或混合版本组合不进入自定义路径。

### 在推理中启用算子

通过 vLLM `additional_config` 启用插件。`strict=true` 表示插件或原生
算子加载失败时直接报错。

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-8B",
    tensor_parallel_size=1,
    max_model_len=8320,
    max_num_batched_tokens=2048,
    enforce_eager=True,
    additional_config={
        "trianglemix": {
            "enabled": True,
            "layers": "5,7,10,15-35",
            "strict": True,
            "stats_log_interval": 100,
        }
    },
)

outputs = llm.generate(
    ["<LONG_CONTEXT>\n\n<QUESTION>"],
    SamplingParams(max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

也可使用兼容环境变量：

```bash
export VLLM_PLUGINS=ascend,trianglemix
export VLLM_ASCEND_ENABLE_TRIANGLE_MIX=1
export VLLM_ASCEND_TRIANGLE_MIX_LAYERS=5,7,10,15-35
export VLLM_ASCEND_TRIANGLE_MIX_STRICT=1
```

手工导入官方 Ascend 模块时应保持以下顺序：

```python
import vllm_ascend.ops
import vllm_ascend.attention.attention_v1
```

### 运行测试

以下命令均从项目根目录执行。安装待验证 wheel 后设置：

```bash
export WHEEL=/absolute/path/to/vllm_ascend_trianglemix-0.1.0-cp310-cp310-linux_aarch64.whl
export MODEL=/absolute/path/to/Qwen3-8B
export RESULTS=/absolute/path/to/new-results
export PROMPT_SOURCE=/absolute/path/to/prompt_source.py
mkdir -p "$RESULTS"
```

依次运行 installed-wheel 正确性、短序列 crossover、模型场景和端到端
TTFT：

```bash
python -m release_validation.run installed-correctness \
  --wheel "$WHEEL" \
  --output "$RESULTS/installed-wheel-correctness.json"

python -m release_validation.run installed-crossover \
  --wheel "$WHEEL" \
  --correctness-report "$RESULTS/installed-wheel-correctness.json" \
  --lengths "512,649,650,1024,2260,2261,8192,8193,8320" \
  --chunk-sizes "512,2048" \
  --output "$RESULTS/installed-wheel-crossover.json"

python -m release_validation.run model-smoke \
  --wheel "$WHEEL" \
  --model "$MODEL" \
  --layers "5,7,10,15-35" \
  --output "$RESULTS/model-smoke.json"

python -m release_validation.run ttft-abba \
  --wheel "$WHEEL" \
  --model "$MODEL" \
  --legacy-script "$PROMPT_SOURCE" \
  --layers "5,7,10,15-35" \
  --runner-args-json '["--runs","6","--long-warmup-runs","1","--enforce-eager"]' \
  --min-gain-percent 9 \
  --output "$RESULTS/ttft-abba.json"
```

`model-smoke` 覆盖 eager、graph、prefix cache、chunked prefill、持续
decode 和 B=1/2/4/8/16 并发。`ttft-abba` 使用 4 个 D-S-S-D cycle，每个
变体均在独立新进程中运行，避免初始化状态和测试顺序污染。

## 5. 算子效率数据

以下结果来自上述 Ascend 910B3/CANN 9.0.1/Qwen3-8B 单卡环境。注意力路径
使用 NPU Event 做相邻 AB/BA 配对，比较官方 dense paged FIA 与带自动
fallback 的 TriangleMix 路由：

| Prompt 长度 / scheduler chunk | 注意力路径耗时降低 |
| --- | ---: |
| 2869 / 512 | 5.227% |
| 4096 / 512 | 14.893% |
| 6144 / 512 | 34.325% |
| 8192 / 512 | 45.587% |
| 8320 / 2048 | 51.158% |

端到端 TTFT 使用发布 wheel 和独立进程 D-S-S-D 配对：

| 指标 | 结果 |
| --- | ---: |
| Dense 平均 TTFT | 0.796091 s（48 samples） |
| TriangleMix 平均 TTFT | 0.724074 s（48 samples） |
| TTFT 降低 | 9.0463% |
| 95% cycle-bootstrap 区间 | 8.9305%～9.1908% |
| 实验规模 | 4 cycles，16 个唯一进程 |

注意力微基准用于判断算子和 crossover 路由效率，不能替代端到端 TTFT。
