// Implements: include/ninfer/ops/argmax.h
// Match: validated contiguous BF16 logits and I32 output.
// Algorithm assumptions: one tile uses a direct reduction; larger domains use
// zero-initialized atomic winners across route-selected row tiles.
#include "ops/launcher/argmax.h"

#include "ninfer/ops/allreduce.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/argmax.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int tiled_block_for(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

} // namespace

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    if (t_count == 0) { return; }

    constexpr int kTileElems = kArgmaxBlock * kArgmaxItemsPerThread;
    const int tiled_blocks   = div_up(valid_rows, kTileElems);
    if (tiled_blocks < 2) {
        argmax_kernel<<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    argmax_tiled_atomic_launch(logits, out, valid_rows,
                               tiled_block_for(physical_rows, valid_rows, t_count), stream);
}

namespace {

void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * kArgmaxItemsPerThread);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        argmax_tiled_atomic_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace

void argmax_local_tp2_launch(const std::array<Tensor, 2>& part,
                             const std::array<Tensor, 2>& out_tokens,
                             std::int32_t valid_rows,
                             const std::array<void*, 2>& local_scalars,
                             const std::array<void*, 2>& peer_scalars,
                             const ::ninfer::ExecutionContext& ec,
                             const PeerEvents& events) {
    const std::int32_t columns = part[0].ne[1];
    if (columns == 0) { return; }

    const std::int32_t shard_rows_0 = part[0].ne[0];
    const std::int32_t valid_0      = std::min(shard_rows_0, valid_rows);
    const std::int32_t valid_1      = std::max(0, valid_rows - shard_rows_0);

    const std::size_t scalar_bytes =
        static_cast<std::size_t>(columns) * sizeof(LocalArgmaxScalar);

    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));

    // Phase A: both ranks compute local argmax on their owned shard and record inputs_ready
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local    = *ec.dev[rank];
        const std::int32_t rank_valid = (rank == 0) ? valid_0 : valid_1;
        const std::int32_t base_index = (rank == 0) ? 0 : shard_rows_0;

        CUDA_CHECK(cudaSetDevice(local.device));
        argmax_local_shard_kernel<<<static_cast<unsigned int>(columns), kArgmaxBlock, 0,
                                    local.stream>>>(
            static_cast<const __nv_bfloat16*>(part[rank].data),
            static_cast<LocalArgmaxScalar*>(local_scalars[rank]), rank_valid, part[rank].ne[0],
            base_index);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(rank), local.stream));
    }

    // Phase B: both ranks wait for peer's inputs_ready, pull peer's 8-byte scalar, and record pull_done
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CUDA_CHECK(cudaSetDevice(local.device));
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.inputs_ready(1 - rank), 0));
        CUDA_CHECK(cudaMemcpyAsync(peer_scalars[rank], local_scalars[1 - rank], scalar_bytes,
                                   cudaMemcpyDeviceToDevice, local.stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(rank), local.stream));
    }

    // Phase C: both ranks resolve the winner locally into out_tokens[rank], and wait on pull_done
    for (int rank = 0; rank < 2; ++rank) {
        const DeviceContext& local = *ec.dev[rank];
        CUDA_CHECK(cudaSetDevice(local.device));
        CUDA_CHECK(cudaStreamWaitEvent(local.stream, events.pull_done(1 - rank), 0));
        constexpr int kResolveBlock = 32;
        const int resolve_blocks    = div_up(columns, kResolveBlock);
        argmax_resolve_peers_kernel<<<static_cast<unsigned int>(resolve_blocks), kResolveBlock, 0,
                                       local.stream>>>(
            static_cast<const LocalArgmaxScalar*>(local_scalars[rank]),
            static_cast<const LocalArgmaxScalar*>(peer_scalars[rank]),
            static_cast<std::int32_t*>(out_tokens[rank].data), columns);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaSetDevice(previous_device));
}

} // namespace ninfer::ops::detail
