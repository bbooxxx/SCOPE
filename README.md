# SCOPE v5

SCOPE 是构建在 DESTINY 之上的三级异质缓存行为级评估器。每层可独立选择器件、容量、相联度、bank、sense amplifier（SA）和 M3D tier 数；C++ 负责 OpenVLA 行为级 load/store trace 与 set-associative LRU 缓存，Python 负责调用 DESTINY、组合延迟/能耗/静态功耗并搜索 `1/(latency × power)` 最优设计。它用于架构筛选，不替代 RTL、SPICE 或真实 GPU trace。

## 1. 构建与运行

```bash
make -j4
make test-scope

# Attention 与 FFN 两次完整搜索
make scope-v5
```

主要目录：

- `component/`：DESTINY 电路组件及新增的可审计 SA 输出。
- `model/`：C++ OpenVLA Attention/FFN 行为 trace 与三级缓存模拟。
- `scope/`：Python 编排、器件库、SA/M3D overlay、功耗与 FoM。
- `config/device_library_v5.json`：v5 单元库；`config/scope_v5.json`：Thor 等面积目标和候选空间。

除本 README 外，仿真、数据和关键点 Markdown 均只保留在本地并由 Git 忽略。

## 2. v5 模型与本轮核查

### Thor 等面积三级缓存

Thor TRM 给出每两个 SM 共享 256 KiB L1 data/shared memory；运行时报告给出 32 MiB GPU L2。SCOPE 保留 256 KiB SRAM L1，把 Thor 32 MiB SRAM L2 的 1/4 面积分给 TFET-eDRAM L2、3/4 分给四层 OSFET-eDRAM L3。按单元库的 4×/16× 密度，目标为：

| 层 | 器件 | SRAM 等面积 | 实际容量 | DESTINY 电路代理 |
|---|---|---:|---:|---:|
| L1 | SRAM | 256 KiB | 256 KiB | 256 KiB |
| L2 | TFET-eDRAM | 8 MiB | 32 MiB | 32 MiB |
| L3 | 4-tier OSFET-eDRAM | 24 MiB | 384 MiB | 512 MiB |

L3 的 512 MiB 只用于满足 DESTINY 的 2 次幂容量约束；命中率、tag 数和单元静态功耗始终使用 384 MiB 架构容量。结构审计会同时报告两者，不把代理容量冒充实际容量。[Thor TRM](https://developer.download.nvidia.com/assets/embedded/secure/jetson/thor/docs/Thor-Soc-TRM_DP-11881-002_v1.1.pdf)、[Thor runtime L2](https://forums.developer.nvidia.com/t/jetson-thor-gpu-stuck-at-1-05ghz/346044)

### 命中率修正

旧版把访问地址压入 24 MiB 抽样工作集，32 MiB L2 因而几乎包住所有复用，这是错误的先验。v5 按 OpenVLA-7B 一层的真实形状推导完整 BF16 地址域：Attention 为 `4H² + 6SH = 148,717,568 B`（141.83 MiB），FFN 为 `3HI + 2SH + 2SI = 288,355,328 B`（275.00 MiB），其中 `S=295, H=4096, I=11008`。C++ 在完整地址域生成 16-byte load/store，先 warmup 4,194,304 次，再统计 4,194,304 次；各级命中率的分母是“真正到达该级的请求”。

该方法与 ChampSim 的 trace/warmup + set-associative 模拟口径一致；NVIDIA 也把 L2 hit rate 定义为到达 L2 的 sector 请求中命中的比例，并指出持久工作集超过可用 L2 时会 thrash。GPU ML 研究进一步报告，大多数 cache hit 来自 kernel 内复用，而非默认可由 L2 容纳整个模型层。[ChampSim](https://github.com/ChampSim/ChampSim/blob/master/docs/src/index.rst)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[MLArchSys/ISCA 2024](https://openreview.net/pdf?id=aYbb7xZuu6)

### SA 与 M3D

原 DESTINY 的 `current sense` 并不是独立电流型 SA：它在同一电压锁存器方程前串入按工艺节点查表的 I–V converter；读出模式由 memory cell 固定，不能按“层 × 器件”搜索。v5 将 converter 延迟、能量和泄漏单独输出并从共同 voltage-latch baseline 中剥离，再只对兼容器件搜索：SRAM/eDRAM 的电压/电荷读出使用 voltage SA，MRAM 的电阻读出可比较 current/voltage SA。当前行为级参数把 current SA 设为 0.6× latency、1.2× dynamic energy、2× gated standby leakage；它表达“更快但功耗更高”的设计方向，所有因子可配置，不宣称为电路签核值。[MRAM current/voltage comparison](https://pure.korea.ac.kr/en/publications/comparative-study-in-response-time-between-a-current-mode-and-a-v)、[STT-MRAM SA review](https://www.sciencedirect.com/science/article/pii/S0026269219300783)、[SRAM SA review](https://www.tandfonline.com/doi/abs/10.4103/0256-4602.107343)

M3D 评估器与器件名称解耦，任意层均可配置 tier 数。其平均垂直 hop 为 `(N-1)/2`，每 hop 延迟用 `0.69(Rdriver+Rvia/2)Cvia`，能量用 `bits·Cvia·Vdd²`，面积用 `via_count·pitch²`。默认 MIV 为 `R=5.5 Ω, C=0.1 fF, pitch=0.1 μm`；模型只覆盖通孔 RC/动态能量/keep-out footprint，不覆盖热、供电和制造良率。[MIV 参数](https://iacomaweb.web.engr.illinois.edu/iacoma-papers/isca19_1.pdf)、[DAC 2023 M3D review](https://www.gtcad.gatech.edu/www/papers/dac23-lingjun.pdf)、[IEDM 2024 archive](https://ieee-iedm.org/wp-content/uploads/2026/05/IEDM2024Archive.pdf)

## 3. 已验证结果

真实 workload 测得读比例为 Attention 95.07%、FFN 94.92%，不再固定为 3:1。SRAM 泄漏按工作簿的 27.5 pW/bit 计算，并显式加入 gated SA 和 SRAM tag 泄漏；目标设计的总静态功耗为 8.601 mW。

| workload | 完整工作集 | 条件 `R1 / R2 / R3` | LPDDR 到达率 | 平均延迟 | 平均功耗 | `1/(ns·mW)` |
|---|---:|---:|---:|---:|---:|---:|
| Attention | 141.83 MiB | 0.351541 / 0.200028 / 0.963217 | 0.019081 | 18.065 ns | 24.963 mW | 0.002217 |
| FFN | 275.00 MiB | 0.359844 / 0.106118 / 0.966080 | 0.019410 | 19.423 ns | 25.784 mW | 0.001997 |

两种 workload 的最优候选均为 `SRAM / TFET-eDRAM / OSFET-eDRAM`。以 Attention 为例，若 L3 改成 96 MiB TFET-eDRAM，L3 条件命中率只有 57.80%、片外到达率 21.89%；384 MiB OSFET-eDRAM 将其改善到 96.32% 和 1.91%。因此修正后 L3 的大容量确实发挥作用，而不是由 L2 预先包住复用数据。

LPDDR 保持 v4 的 Jetson AGX Orin 口径：LPDDR5-6400、256-bit、204.8 GB/s，随机闭页行为延迟 67.5 ns，能耗 2.5 pJ/bit。[Orin Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)、[Ramulator2](https://github.com/CMU-SAFARI/ramulator2)、[DOE EES2 Roadmap](https://www.energy.gov/documents/energy-efficiency-scaling-two-decades-research-and-development-roadmap-draft)
