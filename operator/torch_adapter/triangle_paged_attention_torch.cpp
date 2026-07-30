/*
 * Copyright 2026 TriangleMix contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>

#include "aclnn_torch_adapter/op_api_common.h"

namespace trianglemix_reference {

namespace {

void check_inputs(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table)
{
    TORCH_CHECK(query.is_privateuseone(), "query must be an NPU tensor");
    TORCH_CHECK(
        key_cache.is_privateuseone() && value_cache.is_privateuseone() &&
            block_table.is_privateuseone(),
        "key_cache, value_cache, and block_table must be NPU tensors");
    TORCH_CHECK(
        query.scalar_type() == at::ScalarType::BFloat16,
        "query must be BF16");
    TORCH_CHECK(
        key_cache.scalar_type() == at::ScalarType::BFloat16 &&
            value_cache.scalar_type() == at::ScalarType::BFloat16,
        "key_cache and value_cache must be BF16");
    TORCH_CHECK(
        block_table.scalar_type() == at::ScalarType::Int,
        "block_table must be INT32");
    TORCH_CHECK(
        query.device() == key_cache.device() &&
            query.device() == value_cache.device() &&
            query.device() == block_table.device(),
        "all inputs must be on the same NPU device");
    TORCH_CHECK(
        query.is_contiguous() && key_cache.is_contiguous() &&
            value_cache.is_contiguous() && block_table.is_contiguous(),
        "all inputs must be contiguous");
    TORCH_CHECK(
        query.dim() == 3 && query.size(1) == 32 && query.size(2) == 128,
        "query must be [Tq, 32, 128]");
    TORCH_CHECK(
        key_cache.dim() == 4 && key_cache.size(1) == 128 &&
            key_cache.size(2) == 8 && key_cache.size(3) == 128,
        "key_cache must be [pages, 128, 8, 128]");
    TORCH_CHECK(
        value_cache.sizes() == key_cache.sizes(),
        "value_cache shape must equal key_cache shape");
    TORCH_CHECK(
        block_table.dim() == 2 && block_table.size(0) == 1,
        "block_table must be [1, max_pages]");
}

void check_output(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& output)
{
    TORCH_CHECK(output.is_privateuseone(), "out must be an NPU tensor");
    TORCH_CHECK(
        output.device() == query.device(),
        "out must be on the same NPU device as query");
    TORCH_CHECK(
        output.scalar_type() == at::ScalarType::BFloat16,
        "out must be BF16");
    TORCH_CHECK(
        output.sizes() == query.sizes(),
        "out shape must exactly match query");
    TORCH_CHECK(output.is_contiguous(), "out must be contiguous");
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, query);
    at::assert_no_overlap(output, key_cache);
    at::assert_no_overlap(output, value_cache);
    at::assert_no_overlap(output, block_table);
}

}  // namespace

at::Tensor triangle_paged_sparse_attention(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    int64_t query_start,
    int64_t seq_len,
    int64_t prompt_len,
    double scale)
{
    check_inputs(query, key_cache, value_cache, block_table);
    c10::OptionalDeviceGuard device_guard(query.device());

    at::Tensor output = at::empty_like(query);
    constexpr int64_t q_tile = 32;
    constexpr int64_t page_size = 128;
    constexpr int64_t sink_tokens = 8;
    constexpr int64_t local_window = 512;
    constexpr int64_t dense_tail = 128;

    EXEC_NPU_CMD(
        aclnnTrianglePagedSparseAttention,
        query,
        key_cache,
        value_cache,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        scale,
        q_tile,
        page_size,
        sink_tokens,
        local_window,
        dense_tail,
        output);
    return output;
}

at::Tensor& triangle_paged_sparse_attention_out(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    int64_t query_start,
    int64_t seq_len,
    int64_t prompt_len,
    double scale,
    at::Tensor& output)
{
    check_inputs(query, key_cache, value_cache, block_table);
    check_output(query, key_cache, value_cache, block_table, output);
    c10::OptionalDeviceGuard device_guard(query.device());

    constexpr int64_t q_tile = 32;
    constexpr int64_t page_size = 128;
    constexpr int64_t sink_tokens = 8;
    constexpr int64_t local_window = 512;
    constexpr int64_t dense_tail = 128;

    EXEC_NPU_CMD(
        aclnnTrianglePagedSparseAttention,
        query,
        key_cache,
        value_cache,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        scale,
        q_tile,
        page_size,
        sink_tokens,
        local_window,
        dense_tail,
        output);
    return output;
}

int64_t triangle_paged_sparse_attention_workspace_size(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    int64_t query_start,
    int64_t seq_len,
    int64_t prompt_len,
    double scale)
{
    check_inputs(query, key_cache, value_cache, block_table);
    c10::OptionalDeviceGuard device_guard(query.device());
    at::Tensor output = at::empty_like(query);
    constexpr int64_t q_tile = 32;
    constexpr int64_t page_size = 128;
    constexpr int64_t sink_tokens = 8;
    constexpr int64_t local_window = 512;
    constexpr int64_t dense_tail = 128;

    static const auto getWorkspaceSizeFuncAddr =
        GetOpApiFuncAddr(
            "aclnnTrianglePagedSparseAttentionGetWorkspaceSize");
    static const auto initMemAddr =
        GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr =
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr =
        GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(
        getWorkspaceSizeFuncAddr != nullptr,
        "aclnnTrianglePagedSparseAttentionGetWorkspaceSize not found");

    uint64_t workspace_size = 0;
    uint64_t* workspace_size_addr = &workspace_size;
    aclOpExecutor* executor = nullptr;
    aclOpExecutor** executor_addr = &executor;
    InitHugeMemThreadLocal initMemFunc =
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc =
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    ReleaseHugeMem releaseMemFunc =
        reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    auto converted_params = ConvertTypes(
        query,
        key_cache,
        value_cache,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        scale,
        q_tile,
        page_size,
        sink_tokens,
        local_window,
        dense_tail,
        output,
        workspace_size_addr,
        executor_addr);
    auto getWorkspaceSizeFunc =
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
    const auto workspace_status =
        call(getWorkspaceSizeFunc, converted_params);
    TORCH_CHECK(
        workspace_status == 0,
        "call aclnnTrianglePagedSparseAttentionGetWorkspaceSize failed, "
        "detail:",
        aclGetRecentErrMsg());

    ReleaseConvertTypes(converted_params);
    if (releaseMemFunc) {
        releaseMemFunc(nullptr, false);
    }
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    return static_cast<int64_t>(workspace_size);
}

}  // namespace trianglemix_reference

TORCH_LIBRARY(trianglemix_reference, ops)
{
    ops.def(
        "triangle_paged_sparse_attention("
        "Tensor query, Tensor key_cache, Tensor value_cache, "
        "Tensor block_table, int query_start, int seq_len, "
        "int prompt_len, float scale) -> Tensor");
    ops.def(
        "triangle_paged_sparse_attention_workspace_size("
        "Tensor query, Tensor key_cache, Tensor value_cache, "
        "Tensor block_table, int query_start, int seq_len, "
        "int prompt_len, float scale) -> int");
}

TORCH_LIBRARY_IMPL(trianglemix_reference, PrivateUse1, ops)
{
    ops.impl(
        "triangle_paged_sparse_attention",
        &trianglemix_reference::triangle_paged_sparse_attention);
    ops.impl(
        "triangle_paged_sparse_attention_workspace_size",
        &trianglemix_reference::
            triangle_paged_sparse_attention_workspace_size);
}

TORCH_LIBRARY_FRAGMENT(trianglemix, ops)
{
    ops.def(
        "triangle_paged_sparse_attention.out("
        "Tensor query, Tensor key_cache, Tensor value_cache, "
        "Tensor block_table, int query_start, int seq_len, "
        "int prompt_len, float scale, *, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(trianglemix, PrivateUse1, ops)
{
    ops.impl(
        "triangle_paged_sparse_attention.out",
        &trianglemix_reference::triangle_paged_sparse_attention_out);
}
