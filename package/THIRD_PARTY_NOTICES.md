# Third-party notices

## Huawei CANN 9.0.1

The packaged private OPP contains generated operator artifacts and a vendored
header dependency closure from the CANN 9.0.1 BlockSparseAttention
implementation.

- Copyright: Huawei Technologies Co., Ltd.
- License: CANN Open Software License Agreement Version 2.0
- License text: `licenses/CANN-OSL-2.0.txt`
- Intended platform: systems using Huawei AI Processors and/or CANN, subject
  to the license terms

The vendored source files retain their upstream copyright and license
notices. The dependency closure is described in
`_native/opp/vendors/trianglemix/op_impl/ai_core/tbe/trianglemix_impl/dynamic/vendor/README.md`.

## Runtime dependencies not redistributed by this wheel

PyTorch, torch_npu, vLLM, vLLM-Ascend, and the CANN runtime are dynamically
used but are not redistributed in this wheel. Their respective licenses and
compatibility requirements continue to apply.
