#include "core/topology.h"

#include <sstream>

namespace ninfer {

std::string TopologyInfo::description() const {
    std::ostringstream ss;
    switch (topology) {
    case P2PTopology::BidirectionalUVA:
        ss << "Bidirectional Direct P2P (Hardware UVA enabled in both directions)";
        break;
    case P2PTopology::Asymmetric_0_to_1:
        ss << "Asymmetric P2P (GPU 0 can read GPU 1 via direct UVA; GPU 1 uses Pinned-Host PeerMailbox)";
        break;
    case P2PTopology::Asymmetric_1_to_0:
        ss << "Asymmetric P2P (GPU 1 can read GPU 0 via direct UVA; GPU 0 uses Pinned-Host PeerMailbox)";
        break;
    case P2PTopology::HostMailboxFallback:
        ss << "Host Mailbox Fallback (No P2P support; using cacheline-aligned Pinned-Host PeerMailbox)";
        break;
    }
    ss << " [Dev0: sm_" << dev0_sm_arch << " (" << dev0_sm_count << " SMs), "
       << (dev0_total_vram >> 20) << " MiB | "
       << "Dev1: sm_" << dev1_sm_arch << " (" << dev1_sm_count << " SMs), "
       << (dev1_total_vram >> 20) << " MiB]";
    return ss.str();
}

TopologyInfo probe_hardware_topology(const ExecutionContext& ec) {
    TopologyInfo info{};
    if (ec.tp < 1 || !ec.dev[0].has_value()) {
        return info;
    }

    info.dev0_sm_arch    = ec.dev[0]->sm();
    info.dev0_sm_count   = ec.dev[0]->props.multiProcessorCount;
    info.dev0_total_vram = ec.dev[0]->total_vram();

    if (ec.tp < 2 || !ec.dev[1].has_value()) {
        info.topology = P2PTopology::HostMailboxFallback;
        return info;
    }

    info.dev1_sm_arch    = ec.dev[1]->sm();
    info.dev1_sm_count   = ec.dev[1]->props.multiProcessorCount;
    info.dev1_total_vram = ec.dev[1]->total_vram();

    const int dev0 = ec.dev[0]->device;
    const int dev1 = ec.dev[1]->device;

    int forward = 0;
    int reverse = 0;
    (void)cudaDeviceCanAccessPeer(&forward, dev0, dev1);
    (void)cudaDeviceCanAccessPeer(&reverse, dev1, dev0);

    info.can_access_0_to_1 = (forward != 0);
    info.can_access_1_to_0 = (reverse != 0);
    info.is_bidirectional  = (forward != 0 && reverse != 0);
    info.is_asymmetric     = (forward != reverse);

    if (info.is_bidirectional) {
        info.topology = P2PTopology::BidirectionalUVA;
    } else if (info.can_access_0_to_1) {
        info.topology = P2PTopology::Asymmetric_0_to_1;
    } else if (info.can_access_1_to_0) {
        info.topology = P2PTopology::Asymmetric_1_to_0;
    } else {
        info.topology = P2PTopology::HostMailboxFallback;
    }

    return info;
}

} // namespace ninfer
