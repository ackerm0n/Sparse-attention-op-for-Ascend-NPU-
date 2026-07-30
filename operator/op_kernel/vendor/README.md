# Vendored CANN headers

`cann_9_0_1/block_sparse_attention` contains the header dependency closure
used by `triangle_paged_block_mmad.h`.

- Upstream component: CANN 9.0.1 BlockSparseAttention sample/operator source
- Local source snapshot:
  `vendor/cann_9_0_1/block_sparse_attention`
- Reason for vendoring: CANN dynamic-source operator packages copy
  `op_kernel` but not project-level sibling directories. Keeping this closure
  under `op_kernel` makes an installed package independently compilable.
- License: each copied source file retains its original Huawei copyright and
  CANN Open Software License Agreement Version 2.0 notice.

Only headers are copied. The upstream operator entry points are not part of
the TrianglePagedSparseAttention package.
