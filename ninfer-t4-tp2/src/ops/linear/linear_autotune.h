#pragma once

#include "core/device.h"

#include <cstdint>
#include <string>

namespace ninfer::ops {

// Calibrated crossover thresholds between 1D GEMV / low-register SIMT and Tensor Core MMA.
struct CalibratedLinearPlan {
    bool calibrated                  = false;
    std::int32_t mlp_down_small_t    = 16;  // Shard N=5120, K=8704
    std::int32_t mlp_gate_up_small_t = 4;   // Shard N=17408, K=5120
    std::int32_t attn_qkv_small_t    = 8;   // Shard N=7168, K=5120
    std::int32_t attn_out_small_t    = 16;  // Shard N=5120, K=3072

    [[nodiscard]] std::string description() const;
};

// Returns the global active linear dispatch plan.
CalibratedLinearPlan& active_linear_plan();

// Runs a fast startup micro-benchmark (<60ms) across representative shard geometries to determine
// empirical crossover thresholds on the active GPU architecture. If autotune is false, defaults to
// pre-qualified SM75 analytical thresholds.
CalibratedLinearPlan calibrate_linear_dispatch(const ExecutionContext& ec, bool enable_autotune);

} // namespace ninfer::ops
