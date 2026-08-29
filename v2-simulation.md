# SCOPE v2 仿真报告

## 1. 范围与公共输入

本报告记录 3 次已实际完成的行为级仿真：Attention 7³ 自动搜索、FFN 7³ 自动搜索，以及指定 `SRAM / TFET-eDRAM / OSFET-eDRAM` 映射。完整机器可读数据位于 `results/scope_v2_attention.json`、`results/scope_v2_ffn.json`、`results/scope_v2_requested.json`。

复现命令：

```bash
make -j4
python3 -m unittest discover -s tests -v
python3 scope.py config/scope_v2.json --json-output results/scope_v2_attention.json
python3 scope.py config/scope_v2.json --workload ffn --json-output results/scope_v2_ffn.json
python3 scope.py config/scope_v2_requested.json --json-output results/scope_v2_requested.json
```

公共输入：

| 项目 | 值 |
|---|---|
| 器件候选 | STT-MRAM, SOT-MRAM, SRAM, Si-eDRAM, TFET-eDRAM, 2D-eDRAM, OSFET-eDRAM |
| SRAM 基线容量 | L1 32 KiB, L2 1 MiB, L3 4 MiB/core |
| 容量模式 | 按 140 F²/有效密度等面积扩容，实际值量化为 DESTINY 可用的最近 2 次幂 |
| 层参数 | L1 8-way/1 bank; L2 8-way/4 banks; L3 16-way/8 banks; 64 B; LRU |
| NoC | 两跳 ideal crossbar；2 ns + 0.115776 nJ/跳 |
| write policy | write-back + write-allocate；后台写回能耗计入，不计入需求关键延迟 |
| 片外 | 45.812 ns, 10.24 nJ/64 B |
| BER | L1/L2/L3 = 10⁻¹²/10⁻¹⁰/10⁻⁹；上限 10⁻⁸ |
| 命中率 | `1-exp(-k·C/W·fA·fπ)`，上限 95% |
| static 口径 | 只有 SRAM 使用 27.5 pW/bit；MRAM/eDRAM=0；eDRAM 保持功耗归入 refresh |
| 约束 | BER、endurance、refresh≤retention、high variation |

## 2. 两种 workload 的自动选择

### Attention

输入：`read_fraction=0.82`、`request_rate=1.0×10⁹/s`、reuse window=`64 KiB/2 MiB/8 MiB`、Guidance=`1/[(L/10 ns)¹(P/100 mW)¹]`。共评估 343 种器件映射，125 种通过全部约束。

| 层 | 选中器件 | 容量 | 命中率 | 读/写延迟 | 读/写能耗 | DESTINY 内部结构 |
|---|---|---:|---:|---:|---:|---|
| L1 | Si-eDRAM | 128 KiB | 95% | 0.155/2.000 ns | 0.025/0.024 nJ | ReadEDP; bank 32×512; mat 2×1; subarray 4×8 |
| L2 | Si-eDRAM | 4 MiB | 95% | 0.865/2.000 ns | 0.092/0.098 nJ | ReadEDP; bank 8×256; mat 2×2; subarray 256×16 |
| L3 | TFET-eDRAM | 16 MiB | 95% | 1.874/3.000 ns | 0.176/0.190 nJ | ReadEDP; bank 8×512; mat 2×2; subarray 512×16 |

Latency breakdown：

| 组件 | 到达概率 | 原始延迟 | 加权延迟 |
|---|---:|---:|---:|
| L1 access | 1 | 0.487100 ns | 0.487100 ns |
| L1–L2 NoC + L2 | 0.05 | 3.069300 ns | 0.153465 ns |
| L2–L3 NoC + L3 | 0.0025 | 4.076680 ns | 0.010192 ns |
| off-chip | 0.000125 | 45.812 ns | 0.005727 ns |
| **总计** | | | **0.656483 ns** |

Power breakdown：

| 组件 | 功耗 |
|---|---:|
| dynamic L1 | 24.820000 mW |
| dynamic L1–L2 + L2 | 10.442800 mW |
| dynamic L2–L3 + L3 | 0.735740 mW |
| dynamic off-chip demand | 1.280000 mW |
| WB to L2 / L3 / off-chip | 4.275520 / 3.057760 / 20.480000 mW |
| **dynamic total** | **65.091820 mW** |
| static L1/L2/L3 | 0 / 0 / 0 mW |
| refresh L1/L2/L3 | 0.006815744 / 0.218103808 / 0.000013422 mW |
| **total** | **65.316753 mW** |

`Eavg=0.065091820 nJ/request`，经典 `1/(L·P)=0.0233212485 1/(ns·mW)`，归一化 Guidance score=`23.3212485`。约束全部通过；三层寿命写入量为 `1.38586×10¹³ / 4.81201×10¹⁰ / 6.68335×10⁹`，均低于 `10¹⁵`。

### FFN

输入：`read_fraction=0.67`、`request_rate=1.5×10⁹/s`、reuse window=`80 KiB/2.5 MiB/10 MiB`、Guidance=`1/[(L/10 ns)¹(P/100 mW)²]`。共评估 343 种映射，100 种通过全部约束。器件映射与 Attention 相同，但 DESTINY 目标自动切换为 `ReadDynamicEnergy`，内部结构和指标不同。

| 层 | 选中器件 | 容量 | 命中率 | 读/写延迟 | 读/写能耗 | DESTINY 内部结构 |
|---|---|---:|---:|---:|---:|---|
| L1 | Si-eDRAM | 128 KiB | 95% | 0.754/2.000 ns | 0.017/0.017 nJ | ReadDynamicEnergy; bank 8×32; mat 1×2; subarray 64×32 |
| L2 | Si-eDRAM | 4 MiB | 95% | 1.275/2.000 ns | 0.092/0.098 nJ | ReadDynamicEnergy; bank 8×256; mat 2×2; subarray 256×16 |
| L3 | TFET-eDRAM | 16 MiB | 95% | 2.593/3.000 ns | 0.175/0.190 nJ | ReadDynamicEnergy; bank 8×512; mat 2×2; subarray 512×16 |

| 组件 | 到达概率 | 加权延迟 | 功耗 |
|---|---:|---:|---:|
| L1 | 1 | 1.165180 ns | 25.500000 mW |
| L1–L2 + L2 | 0.05 | 0.175713 ns | 15.731700 mW |
| L2–L3 + L3 | 0.0025 | 0.011818 ns | 1.108973 mW |
| off-chip demand | 0.000125 | 0.005727 ns | 1.920000 mW |
| WB to L2/L3/off-chip | — | — | 6.413280 / 4.586640 / 46.080000 mW |
| **dynamic total** | | | **101.340593 mW** |
| static total | | | 0 mW |
| refresh total | | | 0.224932974 mW |
| **总计** | | **1.358437 ns** | **101.565525 mW** |

`Eavg=0.067560395 nJ/request`，经典 `1/(L·P)=0.00724793167`，Guidance score=`7.13621244`。约束全部通过；三层寿命写入量为 `3.81111×10¹³ / 7.21802×10¹⁰ / 1.00250×10¹⁰`。

## 3. 指定 SRAM / TFET-eDRAM / OSFET-eDRAM 映射

该次使用 Attention profile，并显式允许 L3 OSFET high-BTI variation。TFET-eDRAM 密度为 SRAM 4 倍，四层 OSFET-eDRAM 密度为 SRAM 16 倍，因此容量为 32 KiB / 4 MiB / 64 MiB。

| 层 | 器件 | 容量 | 命中率 | 读/写延迟 | 读/写能耗 | DESTINY 内部结构 |
|---|---|---:|---:|---:|---:|---|
| L1 | SRAM | 32 KiB | 63.2121% | 1/1 ns | 0.018/0.016 nJ | ReadEDP; bank 32×64; mat 1×2; subarray 8×8 |
| L2 | TFET-eDRAM | 4 MiB | 95% | 0.865/3 ns | 0.092/0.098 nJ | ReadEDP; bank 8×256; mat 2×2; subarray 256×16 |
| L3 | OSFET-eDRAM | 64 MiB, 4-tier | 95% | 1.833/10 ns | 0.382/0.397 nJ | ReadEDP; bank 8×512×4; mat 2×2; subarray 512×16 |

| 组件 | 到达概率 | 加权延迟 | 功耗 |
|---|---:|---:|---:|
| L1 | 1 | 1.000000 ns | 17.640000 mW |
| L1–L2 + L2 | 0.367879 | 1.195351 ns | 76.833829 mW |
| L2–L3 + L3 | 0.018394 | 0.097544 ns | 9.205742 mW |
| off-chip demand | 0.000920 | 0.042133 ns | 9.417714 mW |
| WB to L2/L3/off-chip | — | — | 4.275520 / 5.127760 / 20.480000 mW |
| **dynamic total** | | | **142.980564 mW** |
| static L1/L2/L3 | | | 0.00720896 / 0 / 0 mW |
| refresh L1/L2/L3 | | | 0 / 0.000003355 / 0.000053687 mW |
| **总计** | | **2.335028 ns** | **142.987830 mW** |

`Eavg=0.142980564 nJ/request`，Guidance score=`2.99508249`。L1/L2/L3 寿命写入量为 `5.54344×10¹³ / 4.81201×10¹⁰ / 1.67084×10⁹`；对应上限 `10¹⁵ / 10¹⁵ / 10¹²`，全部通过。OSFET 的 `10¹²` 是可配置假设，不是 Device Library 的测量数值。

逐指令结果：

| 路径 | latency | dynamic energy | serialized total power |
|---|---:|---:|---:|
| load @L1 | 1.000 ns | 0.018000 nJ | 18.007266 mW |
| store @L2 | 3.953 ns | 0.224776 nJ | 56.869396 mW |
| load @L3 | 6.084 ns | 0.730552 nJ | 120.084847 mW |
| load @off-chip | 50.235 ns | 11.006552 nJ | 219.108530 mW |

### 验证结论

- `make -j4` 通过；仅有 DESTINY 原项目的 2 个 unused-variable warning。
- 8 项单元测试通过，覆盖 7 种器件文件、SRAM 漏电换算、非 SRAM static=0、命中率、Guidance、密度缩放、平均公式和 WB 能耗。
- Attention/FFN 各 343 组全部评估完成；指定映射完成；三份 JSON 都是本次最终功耗口径的实际输出。
- v1 的两个主要问题已消除：非 SRAM 不再承接旧 DESTINY tag leakage，动态功耗明显高于 static；SRAM 基线命中率为 63.21%，高密度容量可达 95% 设定上限。
