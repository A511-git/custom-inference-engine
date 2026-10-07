#pragma once

#include "core/device.h"

#include <cuda_runtime.h>
#include <string>

namespace ninfer {

enum class P2PTopology {
    BidirectionalUVA,    // Both 0->1 and 1->0 enabled; direct peer UVA memory copies
    Asymmetric_0_to_1,   // GPU 0 can read GPU 1; GPU 1 uses Pinned-Host PeerMailbox
    Asymmetric_1_to_0,   // GPU 1 can read GPU 0; GPU 0 uses Pinned-Host PeerMailbox
    HostMailboxFallback  // No P2P; all cross-rank sync via cacheline-aligned PeerMailbox
};

struct TopologyInfo {
    P2PTopology topology = P2PTopology::HostMailboxFallback;
    bool can_access_0_to_1 = false;
    bool can_access_1_to_0 = false;
    bool is_bidirectional = false;
    bool is_asymmetric = false;
    int dev0_sm_arch = 0;
    int dev0_sm_count = 0;
    int dev1_sm_arch = 0;
    int dev1_sm_count = 0;
    std::size_t dev0_total_vram = 0;
    std::size_t dev1_total_vram = 0;

    [[nodiscard]] std::string description() const;
};

// Probes directional peer access and device capabilities across the ExecutionContext.
TopologyInfo probe_hardware_topology(const ExecutionContext& ec);

} // namespace ninfer
