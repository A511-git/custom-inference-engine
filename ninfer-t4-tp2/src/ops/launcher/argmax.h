#pragma once

// ninfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <array>
#include <cuda_runtime.h>

namespace ninfer {
struct ExecutionContext;
namespace ops {
class PeerEvents;
} // namespace ops
} // namespace ninfer

namespace ninfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

void argmax_local_tp2_launch(const std::array<Tensor, 2>& part,
                             const std::array<Tensor, 2>& out_tokens,
                             std::int32_t valid_rows,
                             const std::array<void*, 2>& local_scalars,
                             const std::array<void*, 2>& peer_scalars,
                             const ExecutionContext& ec,
                             const PeerEvents& events);

} // namespace ninfer::ops::detail
