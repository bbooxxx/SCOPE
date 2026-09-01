![SCOPE — Cache Systems for Embodied AI](assets/scope-banner.png)

# SCOPE v8

SCOPE 是构建在 DESTINY 之上的三级异质缓存行为级评估器。每层可独立配置器件、容量、相联度、bank、物理 subarray 上限、sense amplifier（SA）和 M3D tier 数；C++ 生成 OpenVLA Attention/FFN 的 load/store 行为 trace 并模拟 set-associative LRU 缓存，Python 组合 DESTINY 电路结果、阵列非理想效应、NoC、LPDDR、功耗和 `1/(latency×power)`。v8 在 v7 Evaluation suite 上增加 OSFET-BTI 等价 retention、行维护功耗/带宽开销，并修正了 16 B ISA 访存在 128 B cache line 内的空间局部性和 Attention/FFN tile 复用。

## 1. 构建与运行

```bash
make clean && make -j4
make test-scope
make scope-v8
```

- `component/`：DESTINY 电路组件；v6 新增物理 subarray 行/列约束以及 RWL/RBL 电阻输出。
- `model/`：C++ OpenVLA Attention/FFN tile 调度、16 B load/store 和三级缓存模拟。
- `scope/`：Python 编排；`core.py` 负责开关、组合比较和端到端路径，`bti.py`、`nonideal.py`、`sense_amp.py`、`m3d.py` 分别负责 BTI、阵列非理想、SA 和三维互连。
- `config/scope_v8.json`：v8 目标系统、功能开关和比较矩阵；`config/device_library_v8.json` 在 v7 单元库上增加 OSFET-BTI 模型。

除本 README 外，仿真、关键点和比较 Markdown 只保留本地并由 Git 忽略。机器可读的完整比较可用：

```bash
python3 scope.py config/scope_v8.json --compare \
  --json-output results/scope_v8_attention.json
python3 scope.py config/scope_v8.json --workload ffn --compare \
  --json-output results/scope_v8_ffn.json
```

## 2. Evaluation 开关

`features` 中的三个布尔量可独立修改：

```json
{
  "array_nonideal": true,
  "configurable_peripherals": true,
  "m3d": true
}
```

- `array_nonideal=false`：不叠加 IR drop、2T0C 耦合/串扰、MRAM 线阻和 read-disturb；其他输入不变。
- `configurable_peripherals=false`：停止 SCOPE 的 SA 拓扑重参数化与用户 peripheral override，回退到 DESTINY 原生 SA latency/energy/leakage，不会把外围电路删除或功耗置零。
- `m3d=false`：不叠加 vertical-link latency/energy/footprint；保持缓存容量不变，用于隔离 M3D 模型开销，不代表重新设计一套小容量单层缓存。

## 3. 底层建模

### 阵列非理想

DESTINY 先选择 bank/mat/subarray，并输出实际 `Nrow/Ncol`、RWL/RBL 总电阻。Gain-cell eDRAM 使用：

```text
Ixtalk = Ion - (Nrow-1)Iunsel
ΔVRWL = Ireal·RWL/2
ΔVRBL = Ireal·RBL
treal = CRBL·(ΔVsense+ΔVRBL)/Ireal
PIR = Ireal²·(RWL/2+RBL)
```

RWL 抬高远端 source/VSS，降低读管 `VGS`；模型用可配置 alpha-power 关系降额电流。2T0C 的 RWL/RBL/WWL 耦合严格按各自耦合电容占总 SN 电容的比例计算；互补 N/P 写读管允许用 mismatch fraction 表示上下拉抵消，n-only OSFET 不启用抵消。读串扰按题设 50 mV 偏置下的 `Iunsel` 逐行累加。有效 BER 用 `Q(Vmargin/σSA)` 做行为级估计，因此电流型和电压型 SA 的输入噪声/offset 可以分开配置。[2T gain-cell coupling](https://ieeexplore.ieee.org/document/10704707)、[16-nm gain-cell eDRAM](https://ieeexplore.ieee.org/document/9131838)

MRAM 使用同一条线阻路径求实际读电流和 `I²R`。STT read disturb 使用热激活翻转概率；若超过每次读取上限，限制的是**读电流**并增加读延迟，写电流/写延迟不变。降低写电流不能缓解一次 read disturb。[STT read-disturb model](https://people.ece.umn.edu/groups/VLSIresearch/papers/2014/DATE14_STTMRAM.pdf)、[STT-MRAM reliability](https://publikationen.bibliothek.kit.edu/1000070974/4207942)

### OSFET-BTI 与等价 retention

v8 用可校准的 stretched-exponential 行为模型，而不把 BTI 误当成电荷泄漏：

```text
ΔVth(t) = A·[1-exp(-(d·t/τ)^β)]
ΔVth(1000 s) = 18 mV
Tret,BTI = min{t | ΔVth(t)=70 mV}
AF_T = exp[Ea/k·(1/Tref-1/Top)]
AF_E = exp[γ·(Eop-Eref)]
Tret,op = Tret,BTI/(AF_T·AF_E)
Tref = 0.8·Tret,op
Pref = Nrow·Eref,row/Tref
occupancy = rows_per_bank·(Lread+Lwrite)/Tref
```

300 K、2.5 MV/cm、1 ks 的 ITO:F 实验点 `ΔVth=18 mV` 用于校准电压尺度；近年 oxide-TFT 文献给出 `β=0.4–0.63`、`τ=1.4×10⁴–8.2×10⁵ s` 的范围，v8 取 `β=0.5`、`τ=7.5×10⁵ s`。参考条件下 `Vth,max=70 mV` 对应 `16913.924 s`；再使用 `Ea=0.95 eV` 的 Arrhenius 温度律和显式 E-model，映射到 `368 K / 5 MV/cm` 的高温高场 cache stress corner，得到运行等价 retention `22.272 ms`、维护间隔 `17.817 ms`。间隔末 `ΔVth=63.096 mV`，cell-read latency guardband 为 `1.08842×`。其中场加速系数 `γ=2.7 cm/MV` 是公开标注的 SCOPE stress-corner 校准参数，不是某个已流片 cache macro 的寿命量测。[2024 VLSI ITO:F reliability](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631418)、[2025 oxide-TFT time law](https://doi.org/10.1002/aelm.202400766)、[IGZO time–temperature model](https://doi.org/10.1063/1.3580611)、[95 °C oxide-TFT stress context](https://doi.org/10.1002/sstr.202300375)

OSFET 每次维护为“整行读出+重写/恢复”。因此 v8 同时累加读写能量和 bank busy time，不只增加一个静态功耗常数。

### Attention/FFN 空间局部性与 tile 复用

v7 对 16 B ISA 向量访存进行伪随机地址抽样，会破坏 128 B line 内结构；早期 v8 又把 line 内 8 个向量全部当成独立 cache probe，使 L1 命中率被 7 个后续向量人为抬高到约 94%。当前 v8 保留 16 B ISA 读写统计，但将同一 line 的 `8×16 B` 合并成一次 128 B cache transaction；524,288 个 measured transaction 对应 4,194,304 个 ISA 向量。Attention 使用 `16×64×32` FlashAttention tile，FFN 使用 `16×128×32` GEMM tile，完整权重地址域和完整工作集均保留。[OpenVLA model config](https://github.com/openvla/openvla/blob/main/prismatic/conf/models.py)、[FlashAttention](https://arxiv.org/abs/2205.14135)

### SA、M3D 与 NoC 修正

Gain-cell 的 SN 控制独立读管，最终产生 RBL 放电电流，所以并非“只能 voltage sense”。电压型 SA 等待 `CRBL·ΔV/I` 形成电压差，输入阻抗高、参考和静态电流较简单；电流型 SA 用低输入阻抗钳位 RBL 并直接比较 cell/reference current，通常能更早判决、减小 RBL swing，但付出偏置功耗、current-mirror/reference mismatch。谁的准确率更高并非固定结论，取决于 `signal / input-referred noise`。v6 允许所有 gain-cell eDRAM 搜索两类 SA，并给 latency、energy、noise 三组独立参数。[Current-sense low-input-impedance principle](https://globals.ieice.org/en_transactions/electronics/10.1587/e79-c_8_1120/_p)、[current/voltage SA comparison](https://doi.org/10.1109/ICECA.2019.8822122)

DESTINY 原有 `currentSense` 实际是“查表 I–V converter + 同一个 voltage latch”：`SenseAmp.cpp` 给 converter 加固定延迟/能耗，随后仍执行 `tau·log(Vdd/Vsense)`。v6 会先剥离 converter，再施加独立 current/voltage 拓扑参数与准确率模型；因此它不再把 converter 开销误当成真正电流型 SA 本身。

M3D 同时报告裸 MIV Elmore RC 与可驱动 hop。默认 `Rvia=5.5 Ω, Cvia=0.1 fF` 的裸 RC 只有 `0.00002098 ns`（4 tier 平均 1.5 hops），但 45-nm 4× inverter 驱动的 MIV 报告约 40 ps/hop，因此关键路径取二者较大值，4-tier L3 为 60 ps，而不是沿用不现实的 0.021 ps。能量包含 `Cvia·Vdd²` 与可配置接口能耗，面积按 landing/keep-out pitch 计数。[MIV physical parameters](https://iacomaweb.web.engr.illinois.edu/iacoma-papers/isca19_1.pdf)、[driven MIV delay](https://web.ece.ucsb.edu/~iakgun/files/DAC2019.pdf)

NoC 延迟按每方向 `hops·(router cycles+link cycles)/clock`，分别列出 64-bit 请求和 1024-bit cache-line 响应；能量分成 router 与 `pJ/bit/mm` 走线项。v8 用 `1.75 router cycles + 1.5 link cycles @ 1 GHz`，因此每方向 3.25 ns、一次 request+response 为 6.5 ns。当前单请求 demo 不模拟排队，因此这不是拥塞上界。NVIDIA 官方文档说明 GPU 全局存储使用 32/64/128 B 对齐事务，并记录了 128 B cache line；v8 因此用 128 B 作为统一的事务粒度，但不假定真实 GPU 每个物理层级的 sector 都完全相同。[CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf)、[BookSim](https://crd.lbl.gov/assets/pubs_presos/booksimispass.pdf)、[ORION 2.0](https://escholarship.org/uc/item/5jd3c1gv)

## 4. v8 已验证结果

Thor 等面积目标保持 `256 KiB SRAM / 32 MiB TFET-eDRAM / 384 MiB 4-tier OSFET-eDRAM`。v5 的 OSFET `4096×1024` 是 DESTINY 在只限制“每 mat 有几个 subarray”、却不限制物理行长时选出的 EDP 解；v6 在搜索阶段加入 `Nrow≤1024`，重新得到 `1024×512`，不是事后缩放。

| workload | 实测读比例 | 条件 `R1 / R2 / R3` | LPDDR 到达率 | 平均延迟 | 平均功耗 | `1/(ns·mW)` |
|---|---:|---:|---:|---:|---:|---:|
| Attention | 95.055% | 0.475578 / 0.920167 / 0.883417 | 0.48809% | 7.08472 ns | 15.91428 mW | 0.00886932 |
| FFN | 94.913% | 0.512907 / 0.858221 / 0.929351 | 0.48790% | 7.67733 ns | 16.00067 mW | 0.00814052 |

Attention 中，全 SRAM 为 `21.31530 ns / 13.23904 mW / 0.00354366`，全 OSFET-eDRAM 为 `11.81002 ns / 21.20979 mW / 0.00399221`，最优异质组合为 `7.08472 ns / 15.91428 mW / 0.00886932`；异质 latency 分别降低 66.76% 和 40.01%。FFN 的三组结果依次为 `24.79679/13.65952/0.00295236`、`11.50217/21.17694/0.00410541`、`7.67733/16.00067/0.00814052`；异质 latency 分别降低 69.04% 和 33.25%。

异质 64-bank L3 的 BTI 维护功耗为 `3.68154 mW`，带宽占用率为 `21.36%`；全 OSFET 三层合计为 `4.92785 mW`，L1/L2/L3 带宽占用依次为 `62.66% / 24.39% / 21.36%`。因此 refresh 会同时影响正常访存 latency 和 power，而不再是可忽略小数。

以 Attention 异质组合为基线：关闭阵列非理想后延迟从 `7.08472 ns` 降至 `7.02262 ns`；关闭 M3D 后降至 `7.08152 ns`。关闭可配置外围电路后回到 DESTINY 原生 SA leakage，总功耗从 `15.91428 mW` 增至 `849.36855 mW`。v8 内置验收检查会要求异质 latency 为 7–8 ns、相对两种纯配置均改善至少 30%、L1 hit rate 不高于 60%且全 OSFET refresh 不可忽略。28 项单元测试和 Attention/FFN 完整比较均通过。

LPDDR 使用 Jetson AGX Orin 口径的 LPDDR5-6400：256-bit、204.8 GB/s、随机闭页行为延迟 67.5 ns、2.5 pJ/bit。[Orin Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)、[Ramulator2](https://github.com/CMU-SAFARI/ramulator2)
