![SCOPE — Cache Systems for Embodied AI](assets/scope-banner.png)

# SCOPE v7

SCOPE 是构建在 DESTINY 之上的三级异质缓存行为级评估器。每层可独立配置器件、容量、相联度、bank、物理 subarray 上限、sense amplifier（SA）和 M3D tier 数；C++ 生成 OpenVLA Attention/FFN 的 load/store 行为 trace 并模拟 set-associative LRU 缓存，Python 组合 DESTINY 电路结果、阵列非理想效应、NoC、LPDDR、功耗和 `1/(latency×power)`。v7 增加可重复的 Evaluation suite：在保持 trace、容量、缓存策略、NoC 和 LPDDR 不变的前提下，独立开关阵列非理想、可配置外围电路和 M3D 开销，并自动比较全 SRAM、全 OSFET-eDRAM 和可行最优异质组合。统一 cache line 为 128 B。

## 1. 构建与运行

```bash
make clean && make -j4
make test-scope
make scope-v7
```

- `component/`：DESTINY 电路组件；v6 新增物理 subarray 行/列约束以及 RWL/RBL 电阻输出。
- `model/`：C++ OpenVLA Attention/FFN 行为 trace 与三级缓存模拟。
- `scope/`：Python 编排；`core.py` 负责开关、组合比较和端到端路径，`nonideal.py`、`sense_amp.py`、`m3d.py` 分别负责阵列非理想、SA 和三维互连。
- `config/scope_v7.json`：v7 目标系统、功能开关和比较矩阵；`config/device_library_v7.json` 继承 v6 数值并固定消融语义。

除本 README 外，仿真、关键点和比较 Markdown 只保留本地并由 Git 忽略。机器可读的完整比较可用：

```bash
python3 scope.py config/scope_v7.json --compare \
  --json-output results/scope_v7_attention.json
python3 scope.py config/scope_v7.json --workload ffn --compare \
  --json-output results/scope_v7_ffn.json
```

## 2. v7 Evaluation 开关

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

### SA、M3D 与 NoC 修正

Gain-cell 的 SN 控制独立读管，最终产生 RBL 放电电流，所以并非“只能 voltage sense”。电压型 SA 等待 `CRBL·ΔV/I` 形成电压差，输入阻抗高、参考和静态电流较简单；电流型 SA 用低输入阻抗钳位 RBL 并直接比较 cell/reference current，通常能更早判决、减小 RBL swing，但付出偏置功耗、current-mirror/reference mismatch。谁的准确率更高并非固定结论，取决于 `signal / input-referred noise`。v6 允许所有 gain-cell eDRAM 搜索两类 SA，并给 latency、energy、noise 三组独立参数。[Current-sense low-input-impedance principle](https://globals.ieice.org/en_transactions/electronics/10.1587/e79-c_8_1120/_p)、[current/voltage SA comparison](https://doi.org/10.1109/ICECA.2019.8822122)

DESTINY 原有 `currentSense` 实际是“查表 I–V converter + 同一个 voltage latch”：`SenseAmp.cpp` 给 converter 加固定延迟/能耗，随后仍执行 `tau·log(Vdd/Vsense)`。v6 会先剥离 converter，再施加独立 current/voltage 拓扑参数与准确率模型；因此它不再把 converter 开销误当成真正电流型 SA 本身。

M3D 同时报告裸 MIV Elmore RC 与可驱动 hop。默认 `Rvia=5.5 Ω, Cvia=0.1 fF` 的裸 RC 只有 `0.00002098 ns`（4 tier 平均 1.5 hops），但 45-nm 4× inverter 驱动的 MIV 报告约 40 ps/hop，因此关键路径取二者较大值，4-tier L3 为 60 ps，而不是沿用不现实的 0.021 ps。能量包含 `Cvia·Vdd²` 与可配置接口能耗，面积按 landing/keep-out pitch 计数。[MIV physical parameters](https://iacomaweb.web.engr.illinois.edu/iacoma-papers/isca19_1.pdf)、[driven MIV delay](https://web.ece.ucsb.edu/~iakgun/files/DAC2019.pdf)

NoC 延迟按每方向 `hops·(router cycles+link cycles)/clock`，分别列出 64-bit 请求和 1024-bit cache-line 响应；能量分成 router 与 `pJ/bit/mm` 走线项。当前单请求 demo 不模拟排队，因此这不是拥塞上界。NVIDIA 官方文档说明 GPU 全局存储使用 32/64/128 B 对齐事务，并记录了 128 B cache line；v6 因此用 128 B 作为统一的架构粒度，但不假定真实 GPU 每个物理层级的 sector 都完全相同。[CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf)、[BookSim](https://crd.lbl.gov/assets/pubs_presos/booksimispass.pdf)、[ORION 2.0](https://escholarship.org/uc/item/5jd3c1gv)

## 4. v7 已验证结果

Thor 等面积目标保持 `256 KiB SRAM / 32 MiB TFET-eDRAM / 384 MiB 4-tier OSFET-eDRAM`。v5 的 OSFET `4096×1024` 是 DESTINY 在只限制“每 mat 有几个 subarray”、却不限制物理行长时选出的 EDP 解；v6 在搜索阶段加入 `Nrow≤1024`，重新得到 `1024×512`，不是事后缩放。

| workload | 实测读比例 | 条件 `R1 / R2 / R3` | LPDDR 到达率 | 平均延迟 | 平均功耗 | `1/(ns·mW)` |
|---|---:|---:|---:|---:|---:|---:|
| Attention | 95.074% | 0.359391 / 0.204709 / 0.963898 | 1.8393% | 16.344 ns | 39.071 mW | 0.001566 |
| FFN | 94.926% | 0.363805 / 0.107324 / 0.966551 | 1.8996% | 17.704 ns | 40.968 mW | 0.001379 |

Attention 中，全 SRAM 为 `44.188 ns / 59.531 mW / 0.0003801`；约束内最优异质组合为 `16.344 ns / 39.071 mW / 0.0015659`。全 OSFET 虽有更高原始 FoM `0.0024873`，但 OSFET 的 high BTI variation 违反 L1/L2 的 low-variation 约束，因此标记为不可行，不会被当作最优设计。

以 Attention 异质组合为基线：关闭阵列非理想后延迟下降 0.4095 ns；关闭 M3D 开销后延迟下降 0.0306 ns。关闭可配置外围电路后回到 DESTINY 原生 SA leakage，总功耗从 39.071 mW 增至 872.526 mW；这一项是原生模型与 SCOPE 行为库的消融，不是对实体芯片的量测值。25 项单元测试和 Attention/FFN 完整比较均通过。

LPDDR 使用 Jetson AGX Orin 口径的 LPDDR5-6400：256-bit、204.8 GB/s、随机闭页行为延迟 67.5 ns、2.5 pJ/bit。[Orin Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)、[Ramulator2](https://github.com/CMU-SAFARI/ramulator2)
