![SCOPE](assets/scope-banner.png)

## 1. Overview

SCOPE builds on [DESTINY](https://github.com/sparsh0mittal/destiny_3d_cache) to explore heterogeneous caches for embodied-AI workloads. It connects device and array costs to a three-level cache model, estimating memory access latency, power, conditional hit rates, and FoM = 1/(latency × power) for OpenVLA Attention and FFN workloads.

Configure capacity, device, associativity, banks and sense amplifiers in JSON. The independent `features` switches enable or disable `array_nonideal`, `configurable_peripherals` and `m3d`.

The [Orin budget](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) is one SM-equivalent 192 KiB L1 and a 4 MiB SRAM-area budget split into 1 MiB-equivalent L2 and 3 MiB-equivalent L3; density determines actual capacities. This is a proposed cache hierarchy, not a cycle-accurate Orin GPU.

The C++ trace expands complete projection GEMMs in [grouped CTA order](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), causal FlashAttention and SwiGLU traffic. Register/shared-memory accumulation does not generate global loads. `trace_kind`, `group_m`, `indexing` (modulo or generic XOR folding) and warmup are configurable; the default is a cold operator, with no post-hoc hit-rate correction. This is a serialized behavioral schedule, not a GPU capture.

## 2. Run and example output

Requirements: Python 3.10+ and a C++17 compiler; no third-party Python packages are required.

```bash
make -j2
python3 -m unittest discover -s tests -v
python3 scope.py config/scope_v9.json --compare --json-output results/scope_v9_attention.json
python3 scope.py config/scope_v9.json --workload ffn --compare --json-output results/scope_v9_ffn.json
python3 scripts/v9_reports.py
```

The output is formatted as follows:

```
Best cache configuration:

- L1: SRAM, 192 KiB
- L2: AsyFET-eDRAM, 4 MiB
- L3: OSFET-eDRAM, 48 MiB
- Latency: 3.228609 ns (Atten), 6.558565 ns (FFN)
- Power: 5.700408 mW (Atten), 8.451484 mW (FFN)
- FoM (1/ns/mW): 0.0543348758 (Atten), 0.0180409008 (FFN)
```
