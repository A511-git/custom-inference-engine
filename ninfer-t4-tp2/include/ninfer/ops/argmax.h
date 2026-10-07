#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

struct LocalArgmaxScalar {
    float val;
    std::int32_t idx;
};

class PeerEvents;

/**
 * Local LM-Head Argmax Shortcut for TP2.
 *
 * Slashes PCIe collective transfer from 496 KiB per token down to 8 bytes per token.
 * `part[0]` holds device 0's logits [shard_rows_0, columns];
 * `part[1]` holds device 1's logits [shard_rows_1, columns].
 * `out_tokens[r]` receives the global argmax token IDs [columns].
 */
void argmax_tp2(const std::array<Tensor, 2>& part, const std::array<Tensor, 2>& out_tokens,
                std::int32_t valid_rows, const std::array<void*, 2>& local_scalars,
                const std::array<void*, 2>& peer_scalars, const ExecutionContext& ec,
                const PeerEvents& events);

} // namespace ninfer::ops
