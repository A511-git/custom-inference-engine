# Kaggle 2× Tesla T4 — Complete Hardware & Runtime Information

**Measurement date:** 2026-09-30  
**Purpose:** Reusable baseline record for the current Kaggle notebook environment.

---

## 1. CPU

| Property | Measured value |
|---|---|
| CPU | Intel(R) Xeon(R) CPU @ 2.00GHz |
| Sockets | 1 |
| Physical CPU cores | 2 |
| Threads per core | 2 |
| Logical CPUs | 4 |
| CPU affinity | CPUs 0–3 |
| cgroup CPU quota | `400000 100000` |
| Effective CPU quota | 4 CPUs |
| L3 cache | 38.5 MiB (1 instance) |
| Sustained inference frequency | Not measured |

### CPU interpretation

The environment exposes **2 physical cores / 4 logical threads** and a container quota equivalent to 4 CPUs.

---

## 2. System RAM

| Property | Measured value |
|---|---:|
| Total RAM | 32,870,492 kB |
| Total RAM ≈ | **31.35 GiB** |
| Available RAM | 31,395,376 kB |
| Available RAM ≈ | **29.24 GiB** |
| Swap | **0 kB** |

There is no swap available to the notebook.

---

## 3. GPUs

### GPU 0

| Property | Value |
|---|---|
| Model | Tesla T4 |
| VRAM | 15,360 MiB |
| VRAM | 15 GiB |
| Free VRAM at measurement | 14,912 MiB |
| Free VRAM ≈ | 14.56 GiB |
| PCIe generation | **Gen 3** |
| PCIe width | **x16** |
| Maximum PCIe device link | Gen 3 ×16 |
| PCI bus ID | `00000000:00:04.0` |
| Observed SM clock | 300 MHz |
| Observed memory clock | 405 MHz |
| GPU utilization | 0% |
| Memory utilization | 0% |
| Observed power | 12.71 W |
| Power limit | 70 W |

### GPU 1

| Property | Value |
|---|---|
| Model | Tesla T4 |
| VRAM | 15,360 MiB |
| VRAM | 15 GiB |
| Free VRAM at measurement | 14,912 MiB |
| Free VRAM ≈ | 14.56 GiB |
| PCIe generation | **Gen 3** |
| PCIe width | **x16** |
| Maximum PCIe device link | Gen 3 ×16 |
| PCI bus ID | `00000000:00:05.0` |
| Observed SM clock | 810 MHz |
| Observed memory clock | 5000 MHz |
| GPU utilization | 0% |
| Memory utilization | 0% |
| Observed power | 27.78 W |
| Power limit | 70 W |

### Aggregate GPU memory

- GPU count: **2**
- Nominal total VRAM: **30 GiB**
- Free VRAM at measurement: **29,824 MiB ≈ 29.12 GiB**
- The 30 GiB is **not unified memory**.
- Each T4 has its own separate 15 GiB VRAM.

> The observed clocks above were captured while the GPUs were idle. They are not peak or sustained inference clocks.

---

## 4. GPU Topology

### NVIDIA topology

```text
GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB             0-3
GPU1    PHB      X              0-3
```

### Meaning

- GPU0 ↔ GPU1 topology: **PHB**
- Both GPUs are attached through the PCIe host-bridge path.
- NVLink: **Not present**
- NUMA affinity: **0**
- CPU affinity: **0–3**

---

## 5. GPU Peer-to-Peer Access

P2P is supported in both directions:

| Direction | Status |
|---|---|
| GPU0 → GPU1 | **OK** |
| GPU1 → GPU0 | **OK** |

---

## 6. Measured GPU Transfer Performance

These were measured using the PyTorch transfer benchmark with a large transfer buffer.

| Transfer | Measured |
|---|---:|
| CPU → GPU0 (H2D) | **11.432 GiB/s** |
| GPU0 → CPU (D2H) | **12.188 GiB/s** |
| CPU → GPU1 (H2D) | **11.440 GiB/s** |
| GPU1 → CPU (D2H) | **12.168 GiB/s** |
| GPU0 → GPU1 | **9.177 GiB/s** |
| GPU1 → GPU0 | **9.184 GiB/s** |

These are practical measured transfer rates, not theoretical PCIe bandwidth values.

---

## 7. CUDA / PyTorch / NCCL

| Component | Version |
|---|---|
| CUDA | **12.8** |
| PyTorch | **2.10.0+cu128** |
| NCCL | **2.27.5** |
| GPUs visible to PyTorch | **2** |

---

## 8. Storage Paths

### `/kaggle/working`

```text
Exists: YES
Filesystem: ext4
Backing: /dev/loop2[/kaggle/working]
Read/write: YES
```

| Property | Value |
|---|---:|
| Reported capacity | ~20 GB |
| Reported free space | ~20 GB |
| Used at measurement | ~292 KB |

This location is **far too small for an 82GB model file**.

---

### `/kaggle/temp`

```text
Exists: NO
```

The current Kaggle runtime does **not** provide a separate `/kaggle/temp` directory.

Therefore there is no separate `/kaggle/temp` disk to benchmark or use in this runtime.

---

### `/tmp`

```text
Exists: YES
Filesystem: overlayfs
```

The raw container filesystem report showed:

```text
8.0 TB total
~6.9 TB used
~1.1 TB free
```

However, the practical capacity available to the Kaggle runtime has been observed to be approximately:

> **~100 GB usable**

For planning an 82GB model workload, **use ~100GB as the practical `/tmp` capacity**, not the raw overlay filesystem figure.

Important distinction:

- **Raw overlay capacity:** ~1.1 TB free according to `df`
- **Practical runtime capacity:** ~100 GB observed
- These are not equivalent guarantees.

---

### `/dev/shm`

```text
Filesystem: tmpfs
Capacity: 14 GB
```

| Property | Value |
|---|---:|
| Total | 14 GB |
| Used at measurement | 0 |
| Free at measurement | 14 GB |

This is shared memory, not normal persistent model storage.

---

### `/kaggle/input`

Base `/kaggle/input` mount:

```text
Filesystem: ext4
Backing: /dev/loop2[/kaggle/input]
Read-only: YES
Capacity: ~20 GB
```

There is also a specific NFS-backed dataset path under `/kaggle/input`.

---

## 9. NFS Dataset Storage

Observed dataset path:

```text
/kaggle/input/datasets/newmailserver/test-img
```

Mounted from:

```text
192.168.5.2:/data/kagglesdsdata/...
```

Filesystem:

```text
NFS
```

Properties observed:

- Read-only
- Network-backed
- NFS version 3
- `rsize=524288`
- `wsize=524288`
- TCP transport

This should be considered **network storage**, not equivalent to local disk.

---

## 10. Block Devices

Observed block devices included:

```text
loop0    ~1G
loop1    ~96G
loop2    ~20G

sda      PersistentDisk   ~8T   read-only
sdb      PersistentDisk   ~128G  read/write
```

The `sdb` device contains the system partitions used for CUDA and related system software.

`loop2` is the device associated with the ~20GB Kaggle working/input/lib filesystem.

---

## 11. Important Storage Interpretation

For a large model file:

### `/kaggle/working`
- Writable
- Local ext4 mount
- Only ~20GB
- **Cannot hold an 82GB model**

### `/kaggle/temp`
- Does not exist

### `/tmp`
- Exists
- Overlay filesystem
- Practical capacity observed: **~100GB**
- Potential location for an 82GB model, subject to confirming exact free space and actual read performance

### `/kaggle/input`
- Read-only
- Some dataset subpaths are NFS/network-backed
- Not a good default for a storage-sensitive model benchmark

---

## 12. Storage Benchmark Status

The storage benchmark code was executed for:

```text
/kaggle/working
/kaggle/temp
```

However:

- `/kaggle/working` benchmark numeric read/write results were not present in the pasted console output.
- `/kaggle/temp` does not exist in this runtime.

Therefore no storage throughput number should be assumed from the available data.

The next useful measurement is the **actual read speed of the 82GB model file from `/tmp`**, once the file is placed there.

---

## 13. Current Kaggle Baseline

```text
CPU
├── Intel Xeon @ 2.00 GHz
├── 1 socket
├── 2 physical cores
├── 4 logical CPUs
├── 38.5 MiB L3
└── 4-CPU cgroup quota

RAM
├── ~31.35 GiB total
├── ~29.24 GiB available at measurement
└── 0 swap

GPU
├── Tesla T4 #0
│   ├── 15 GiB VRAM
│   ├── PCIe Gen3 x16
│   └── ~11.43 GiB/s H2D
│
└── Tesla T4 #1
    ├── 15 GiB VRAM
    ├── PCIe Gen3 x16
    └── ~11.44 GiB/s H2D

GPU ↔ GPU
├── P2P: OK
├── Topology: PHB
├── No NVLink
└── ~9.18 GiB/s measured

Software
├── CUDA 12.8
├── PyTorch 2.10.0+cu128
└── NCCL 2.27.5

Storage
├── /kaggle/working → ~20GB writable ext4
├── /kaggle/temp    → does not exist
├── /tmp            → overlayfs, ~100GB practical capacity observed
├── /dev/shm        → 14GB tmpfs
└── /kaggle/input   → ~20GB base read-only mount + NFS dataset subpaths
```

---

## 14. Missing Measurements

The following have **not** been measured yet:

1. RAM type, frequency and channel configuration.
2. Sustained CPU frequency under inference load.
3. Actual `/tmp` sequential read/write throughput.
4. Actual 82GB model-file read throughput from `/tmp`.
5. Random/offset read performance of the model file.
6. Actual CPU utilization during inference.
7. Actual RAM usage during inference.
8. Actual VRAM usage distribution during inference.
9. End-to-end inference performance for the target model.
10. Exact inference software/build and runtime parameters.

---


This document is the current **Kaggle-only hardware/runtime baseline**. It contains measured values only where they were actually obtained and explicitly marks information that remains unknown.
