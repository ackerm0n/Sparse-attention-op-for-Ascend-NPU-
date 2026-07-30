#!/usr/bin/env python3
"""Report the aclnn workspace size without launching the custom kernel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from npu_v2_harness_common import DEFAULT_NPU_LOCK, exclusive_npu_lock


QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
QUERY_START = 521
QUERY_TOKENS = 1
SEQ_LEN = 522
PROMPT_LEN = 2048
SCALE = HEAD_DIM**-0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_NPU_LOCK)
    parser.add_argument("--lock-timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logical_pages = math.ceil(SEQ_LEN / PAGE_SIZE)
    physical_pages = logical_pages + 5
    query = torch.zeros(
        QUERY_TOKENS,
        QUERY_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    cache_shape = (
        physical_pages,
        PAGE_SIZE,
        KV_HEADS,
        HEAD_DIM,
    )
    key = torch.zeros(cache_shape, dtype=torch.bfloat16)
    value = torch.zeros(cache_shape, dtype=torch.bfloat16)
    block_table = torch.tensor([[7, 2, 9, 1, 6]], dtype=torch.int32)

    with exclusive_npu_lock(args.lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        torch.ops.load_library(str(args.adapter.resolve()))
        workspace_size = (
            torch.ops.trianglemix_reference
            .triangle_paged_sparse_attention_workspace_size(
                query.npu(),
                key.npu(),
                value.npu(),
                block_table.npu(),
                QUERY_START,
                SEQ_LEN,
                PROMPT_LEN,
                SCALE,
            )
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "aclnn_workspace_size_bytes": workspace_size,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
