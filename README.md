# SCOPE v2

SCOPE 是构建在 DESTINY 之上的三级片上缓存行为级 demo。它不生成精确访存 trace，而是把三个独立 DESTINY 实例串成 L1–L2–L3，对一条 load/store 和平均 workload 输出端到端 latency、dynamic/static/refresh power 及 breakdown。

v2 支持从应用到硬件的自动选择：工作负载 → 可配置 Guidance/FoM → 每层器件 → 密度换算容量 → DESTINY bank/mat/subarray/外围结构 → 端到端结果。L1/L2/L3 均可使用任意一种库内器件，不绑定 SRAM/eDRAM/MRAM 位置。

## 快速运行

```bash
make -j4
make test-scope

# Attention：7³=343 种器件映射自动选优
python3 scope.py config/scope_v2.json \
  --json-output results/scope_v2_attention.json

# FFN：使用另一组 Guidance
python3 scope.py config/scope_v2.json --workload ffn \
  --json-output results/scope_v2_ffn.json

# 指定 SRAM / TFET-eDRAM / OSFET-eDRAM 映射
make scope-requested
```

## 配置要点

- `workloads.attention/ffn`：读写比例、请求率、算子形状、局部性和应用 Guidance；用 `--workload` 选择。
- `guidance.weights`：任意组合 `latency`、`power`、`energy` 的非负指数；评分为归一化指标幂乘积的倒数。还可设置 `limits` 或显式 `destiny_optimization_target`。
- `capacity.mode`：`density_scaled` 按 `140 F² / 器件有效 F²` 从 SRAM 基线等面积扩容；`fixed` 则直接使用每层 `capacity_bytes`。DESTINY 只支持合法阵列容量，非 2 次幂理想容量会量化到最近 2 次幂，JSON 同时保留理想/实际值。
- `layers[].devices`：可限制某层的候选集；不设时自动遍历全部 7 种。容量、相联度、64 B 行宽、LRU/FIFO/RANDOM、bank、BER、刷新和寿命参数都可改。
- NoC 和 write policy 在选优中固定：两跳 ideal crossbar、write-back + write-allocate。

器件库只包含 `STT-MRAM`、`SOT-MRAM`、`SRAM`、`Si-eDRAM`、`TFET-eDRAM`、`2D-eDRAM`、`OSFET-eDRAM`。每种的 `.cell/.cfg` 位于 `config/devices/`，并包含截图表格中的读/写延迟与能耗、漏电、刷新、寿命、variation、密度和 M3D 指标。

## 当前实测

| workload / 映射 | L1 / L2 / L3 | 平均延迟 | 平均总功耗 | static |
|---|---|---:|---:|---:|
| Attention 自动 | Si-eDRAM / Si-eDRAM / TFET-eDRAM | 0.656483 ns | 65.316753 mW | 0 mW |
| FFN 自动 | Si-eDRAM / Si-eDRAM / TFET-eDRAM | 1.358437 ns | 101.565525 mW | 0 mW |
| 指定 Attention | SRAM / TFET-eDRAM / OSFET-eDRAM | 2.335028 ns | 142.987830 mW | 0.007209 mW |

Attention 使用 `ReadEDP`，FFN 因功耗权重更高自动切换为 `ReadDynamicEnergy`，并得到不同的内部 bank/mat/subarray 结构。指定映射的容量为 32 KiB / 4 MiB / 64 MiB，分别对应 SRAM 的 1×、TFET-eDRAM 的 4×和四层 OSFET-eDRAM 的 16×密度扩容。

完整输入、breakdown、约束和逐指令结果见 [v2-simulation.md](v2-simulation.md) 及 `results/scope_v2_*.json`。所有数据来源另整理在本地 `v2-data.md`，该文件已被 Git 忽略，不会上传。

## 模型边界

这是用于快速比较的行为级 demo，不声称替代 gem5/真实 trace/电路级签核。命中率是容量、workload 复用窗口、替换策略和相联度的可配置函数；容量增大时可超过基线 60–80% 并截止在配置上限 95%。eDRAM 表中的 `leakage=Pref` 只在 refresh 中计一次，不重复算 static；raw DESTINY 漏电仍保留在 JSON 供审计。
