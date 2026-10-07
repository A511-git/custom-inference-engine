CUDA device count: 2

Device 0: Tesla T4
  Compute capability: 7.5
  PCI Bus ID: 00:00.4

Device 1: Tesla T4
  Compute capability: 7.5
  PCI Bus ID: 00:00.5

cudaDeviceCanAccessPeer(0,1): 1
cudaDeviceCanAccessPeer(1,0): 1

Device 0: Tesla T4, CC 7.5
Device 1: Tesla T4, CC 7.5

-----------------------------------------

=== COMPILING ===

=== RUNNING ===
CUDA device count: 2

GPU0: Tesla T4 | CC 7.5
GPU1: Tesla T4 | CC 7.5

P2P 0 -> 1: 1
P2P 1 -> 0: 1

=== GPU0 -> GPU1 ===
Transferred: 5.37 GB
Time: 542.248 ms
Bandwidth: 9.90 GB/s

=== GPU1 -> GPU0 ===
Transferred: 5.37 GB
Time: 541.813 ms
Bandwidth: 9.91 GB/s

========================================
FINAL P2P RESULT
========================================
GPU0 -> GPU1 : 9.90 GB/s
GPU1 -> GPU0 : 9.91 GB/s
P2P 0 -> 1   : SUPPORTED
P2P 1 -> 0   : SUPPORTED