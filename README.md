# TriangleMix：面向 vLLM-Ascend 的单次 CANN Launch 稀疏 Prefill

## 1. 项目总览

本项目在昇腾 NPU 上实现 [TriangleMix](https://arxiv.org/abs/2507.21526)
论文提出的 Triangle 稀疏注意力，并作为 vLLM-Ascend 内置的可选
attention 路径接入。昇腾推理栈目前缺少能够真正跳过稀疏区域计算的
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

## 2. vLLM-Ascend 原生接入

实现以最小差异作为 vLLM-Ascend 的原生可选路径：

- `vllm_ascend/attention/trianglemix.py` 提供配置、计划、fallback 原因和
  性能计数；
- `AscendCommonAttentionMetadata` 传递 scheduler 持有的最终 prompt 长度；
- `AscendAttentionMetadataBuilder` 每个 scheduler step 只生成一次不可变
  TriangleMix 计划；
- `AscendAttentionBackendImpl` 仅在选中层和受支持的 prefill 请求上调用
  自定义路径，其他请求保持官方 FIA；
- `csrc/attention/triangle_paged_sparse_attention` 随 vLLM-Ascend 的标准
  CANN 构建流程编译，并注册为
  `torch.ops._C_ascend.npu_triangle_paged_sparse_attention`。

该路径默认关闭。推荐通过 `additional_config.trianglemix` 显式启用；
环境变量仅作为迁移期兼容接口。

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
- Python 接入必须避免运行时替换类方法：配置、metadata 和 dispatch 均在
  vLLM-Ascend 原生对象生命周期内完成；
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

算子对固定计算几何执行 fail-closed 门禁；不支持的 shape、batch、
并行模式或运行阶段不会进入自定义路径。

### 在推理中启用算子

通过 vLLM `additional_config` 启用原生路径。`strict=true` 表示 CANN
算子 launch 失败时直接报错；默认行为是回退官方 FIA。

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
export VLLM_ASCEND_ENABLE_TRIANGLE_MIX=1
export VLLM_ASCEND_TRIANGLE_MIX_LAYERS=5,7,10,15-35
```

### 运行测试

从 vLLM-Ascend 仓库根目录执行：

```bash
export MODEL=/absolute/path/to/Qwen3-8B
export RESULTS=/absolute/path/to/new-results
mkdir -p "$RESULTS"
pytest -sv tests/ut/attention/a2/test_trianglemix.py
```

路由单测覆盖默认关闭、配置优先级、长序列命中、chunked prefill、
decode/batch/graph fallback、shape 门禁和 `_C_ascend` out-operator 契约。
NPU 回归覆盖算子与 FIA reference 的正确性、短序列 crossover、
prefix cache、持续 decode 和 B=1/2/4/8/16；端到端 TTFT 使用独立进程
D-S-S-D 配对，避免初始化状态和测试顺序污染。

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

端到端 TTFT 使用原生 vLLM-Ascend 路径和独立进程 D-S-S-D 配对：

| 指标 | 结果 |
| --- | ---: |
| Dense 平均 TTFT | 0.796091 s（48 samples） |
| TriangleMix 平均 TTFT | 0.724074 s（48 samples） |
| TTFT 降低 | 9.0463% |
| 95% cycle-bootstrap 区间 | 8.9305%～9.1908% |
| 实验规模 | 4 cycles，16 个唯一进程 |

注意力微基准用于判断算子和 crossover 路由效率，不能替代端到端 TTFT。
