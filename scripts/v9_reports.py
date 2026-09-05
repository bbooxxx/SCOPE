"""Generate local v9 delivery reports and the README from measured JSON results."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"all_sram": "S-S-S", "sram_osfet_osfet": "S-O-O",
          "sram_mram_mram": "S-M-M", "optimized": "S-A-O"}

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in rows])

def size(value):
    return f"{value / 1048576:g} MiB" if value >= 1048576 else f"{value / 1024:g} KiB"

def relative_fom_gain(candidate, baseline):
    if candidate <= 0 or baseline <= 0:
        raise ValueError("FoM values must be positive")
    return candidate / baseline - 1.0

def snippet(path, marker, lines=10):
    source = (ROOT / path).read_text().splitlines()
    start = next(i for i, line in enumerate(source) if marker in line)
    return f"{path}:{start+1}\n\n```python\n" + "\n".join(source[start:start+lines]) + "\n```\n"

def main():
    reports = {name: json.loads((ROOT / f"results/scope_v9_{name}.json").read_text())
               for name in ("attention", "ffn")}
    summary, geometry, goals = [], [], []
    for workload, report in reports.items():
        by_case = {item["case"]: item for item in report["architecture_comparison"]}
        for item in report["architecture_comparison"]:
            summary.append([workload, LABELS[item["case"]],
                *[size(v) for v in item["capacities_bytes"]],
                *[f"{100*v:.3f}%" for v in item["conditional_hit_rates"]],
                f'{item["average_latency_ns"]:.6f}', f'{item["average_power_mw"]:.6f}',
                f'{item["fom_per_ns_mw"]:.9g}',
                *[f'{100*relative_fom_gain(item["fom_per_ns_mw"], by_case[base]["fom_per_ns_mw"]):+.3f}%'
                  for base in ("sram_osfet_osfet", "all_sram")]])
            for layer in report["case_reports"][item["case"]]["layers"]:
                raw = layer["raw_destiny_metrics"]
                geometry.append([workload, LABELS[item["case"]], layer["name"],
                    layer["banks"], raw["bank_organization"], raw["mat_organization"],
                    raw["subarray_size"], size(layer["destiny_model_capacity_bytes"]),
                    layer["destiny_optimization_target"]])
        candidate = by_case["optimized"]
        for base, threshold in (("sram_osfet_osfet", 0.10), ("all_sram", 0.80)):
            reference_fom = by_case[base]["fom_per_ns_mw"]
            gain = relative_fom_gain(candidate["fom_per_ns_mw"], reference_fom)
            target_product = 1 / (reference_fom * (1 + threshold))
            goals.append([workload, "S-A-O / " + LABELS[base],
                          f'{100*gain:+.3f}%', f'>= {100*threshold:.0f}%',
                          "Yes" if gain >= threshold else "No",
                          f'{target_product / candidate["average_power_mw"]:.6f}',
                          f'{target_product / candidate["average_latency_ns"]:.6f}'])
    (ROOT / "comparison.md").write_text(
        table(["Workload", "Configuration", "L1 capacity", "L2 capacity", "L3 capacity",
               "L1 conditional hit", "L2 conditional hit", "L3 conditional hit",
               "Latency (ns)", "Power (mW)", "FoM (1/ns/mW)",
               "FoM gain vs S-O-O", "FoM gain vs S-S-S"], summary) + "\n\n" +
        table(["Workload", "Comparison", "Simulated FoM gain", "Target", "Met",
               "Required SAO latency at unchanged power (ns, <=)",
               "Required SAO power at unchanged latency (mW, <=)"], goals) + "\n\n" +
        table(["Workload", "Configuration", "Level", "Configured banks",
               "DESTINY mats/bank", "Subarrays/mat", "Cells/subarray",
               "DESTINY proxy capacity", "Selected search target"], geometry) + "\n")
    winners = {}
    for workload, report in reports.items():
        winner = report["best_evaluated_architecture"]
        winners[workload] = next(x for x in report["architecture_comparison"] if x["case"] == winner)
    a, f = winners["attention"], winners["ffn"]
    same = a["devices"] == f["devices"]
    example = "Best cache configuration" + (":" if same else " (Attention; FFN differs):") + "\n\n"
    example += "\n".join(f"- L{i+1}: {device}, {size(cap)}"
                         for i, (device, cap) in enumerate(zip(a["devices"], a["capacities_bytes"])))
    example += (f'\n- Latency: {a["average_latency_ns"]:.6f} ns (Atten), {f["average_latency_ns"]:.6f} ns (FFN)'
                f'\n- Power: {a["average_power_mw"]:.6f} mW (Atten), {f["average_power_mw"]:.6f} mW (FFN)'
                f'\n- FoM: {a["fom_per_ns_mw"]:.9g} (Atten), {f["fom_per_ns_mw"]:.9g} (FFN)\n')
    if not same:
        example += "\nFFN configuration: " + ", ".join(
            f"L{i+1} {dev} {size(cap)}" for i, (dev, cap) in enumerate(zip(f["devices"], f["capacities_bytes"]))) + "\n"
    readme = """![SCOPE](assets/scope-banner.png)

## 1. Overview

SCOPE builds on [DESTINY](https://github.com/sparsh0mittal/destiny_3d_cache) to explore heterogeneous caches for embodied-AI workloads. It connects device and array costs to a three-level cache model, estimating average memory-access latency, memory-system power, conditional hit rates, BER and FoM = 1/(latency × power) for OpenVLA-shaped Attention and FFN workloads.

Configure capacity, device, associativity, banks and sense amplifiers in JSON. The independent `features` switches enable or disable `array_nonideal`, `configurable_peripherals` and `m3d`. v9 uses an Elmore MIV ladder, read/write interconnect energy, and staircase footprint. OSFET BTI refresh is disabled. AsyFET-eDRAM uses the existing TFET device parameters.

The [Orin budget](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) is one SM-equivalent 192 KiB L1 and a 4 MiB SRAM-area budget split into 1 MiB-equivalent L2 and 3 MiB-equivalent L3; density determines actual capacities. This is a proposed cache hierarchy, not a cycle-accurate Orin GPU.

Results use synthetic addresses informed by tensor sizes and reuse assumptions, not a measured GPU trace. The inherited configuration exposes its 0.15 L3 OSFET latency scale, 0.88 cross-frame reuse assumption and NoC timing parameters; these are behavioral assumptions. MIV independent conduction current defaults to zero until characterized; nonzero current requires a pulse duration. Array footprint defaults to square bank blocks and accepts explicit dimensions. The reported optimum is among the evaluated candidates.

Original SCOPE contributions use MIT; bundled DESTINY/NVSim/CACTI-3DD retain their licenses (see LICENSE and NOTICE). See CONTRIBUTING for changes and validation.

## 2. Run and example output

Requirements: Python 3.10+ and a C++17 compiler; no third-party Python packages are required.

```bash
make -j2
python3 -m unittest discover -s tests -v
python3 scope.py config/scope_v9.json --compare --json-output results/scope_v9_attention.json
python3 scope.py config/scope_v9.json --workload ffn --compare --json-output results/scope_v9_ffn.json
python3 scripts/v9_reports.py
```

The comparison evaluates S-S-S, S-O-O, S-M-M and S-A-O, plus the three feature ablations. Each architecture evaluates the same L2/L3 search targets (read EDP, read dynamic energy, leakage power) and compatible SAs, selecting the highest feasible system FoM. Generated JSON and Markdown reports stay local. To evaluate one explicit path:

```bash
python3 scope.py config/scope_v9.json --explore --op load --hit-level L3
```

"""
    (ROOT / "README.md").write_text(readme + example)
    revision = "# v9 修改\n\n"
    revision += "Orin：192 KiB L1；4 MiB SRAM 面积按 1:3 分配 L2/L3。AsyFET 沿用原 TFET 参数；OSFET refresh=none。\n\n"
    revision += "功率使用 $P=\\lambda E$，电阻项为 $I^2Rt$；$I$ 仅表示独立导通电流，不能重复计入充放电电流。默认导通电流为 0，等待器件参数。\n\n"
    revision += "Elmore：$\\tau_k=0.69\\sum_{i=1}^{k}(\\sum_{j=1}^{i}R_j)C_i$。\n\n"
    revision += snippet("scope/m3d.py", "    delays = ", 8) + "\n"
    revision += "读写能量分别累加，随后由层访问频率转成功率：\n\n"
    revision += snippet("scope/m3d.py", '        energies[operation] = [', 6) + "\n"
    revision += "台阶面积：$A_{extra}=(H+2NL_{CPP})(W+2NL_{CPP})-HW$；N 为存储 tier 数。底层同平面时路径为 0/1/2/3 跳，独立外围层可配置 bottom_tier_hops=1。\n\n"
    revision += snippet("scope/m3d.py", "    margin = ", 2) + "\n"
    revision += "配置：config/scope_v9.json；完整器件库：config/device_library_v9.json。旧库保留供兼容测试。\n\n"
    revision += "系统 FoM 选优：四种架构的 L2/L3 均搜索 ReadEDP、ReadDynamicEnergy 和 LeakagePower，并比较兼容 SA。\n\n"
    revision += snippet("scope/core.py", '            targets = layer.get(', 11) + "\n"
    revision += "配置文件名包含 line 大小与内容摘要，避免并行测试覆盖；读取已有结果时校验容量、相联度和 line 大小。\n\n"
    revision += snippet("scope/core.py", '    filename = (', 7) + "\n"
    revision += "验证：34 项单元测试；Attention/FFN 四配置和三开关对比。通过公式与开关断言，不把预定排名作为验收条件。L1 条件相同，其命中率也相同。\n\n"
    revision += "FoM 增幅按 FoM_SAO/FoM_reference−1 计算；comparison.md 同时列出相对 S-O-O 的 10% 目标、相对 S-S-S 的 80% 目标，以及固定功率或延迟时需要达到的另一指标。这是目标检查，不修改仿真结果。\n\n"
    revision += "来源：Orin technical brief（README 链接）；M3D-MDA Table 1：R=5.5 Ω、C=0.1 fF、pitch=0.2 μm。H/W 默认从等面积正方形 bank 推导，读写摆幅 0.1/1.2 V 为模型输入。\n\n"
    revision += "保留并公开：L3 0.15 延迟缩放、跨帧 0.88 复用和器件 standby 估算。结果为行为级预测，不是实测。MIT 仅覆盖新增代码，第三方版权保留。\n"
    (ROOT / "revision.md").write_text(revision)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    names = sorted({p for p in names if (ROOT / p).is_file()})
    descriptions = {
        "AccessTrace": "生成按工作集和复用参数构造的访存地址。",
        "CacheModel": "模拟缓存命中、替换和脏数据回写。",
        "Bank": "描述存储 bank 的公共结构。",
        "BankWithHtree": "计算采用 H 树布线的 bank。",
        "BankWithoutHtree": "计算不采用 H 树布线的 bank。",
        "BasicDecoder": "计算基础译码电路。",
        "Comparator": "计算标签比较器。",
        "FunctionUnit": "提供电路面积、延迟与能量的公共接口。",
        "InputParameter": "读取 DESTINY 搜索参数。",
        "Mat": "组合 mat 内的子阵列与外围电路。",
        "MemCell": "读取器件单元参数。",
        "Mux": "计算多路选择电路。",
        "OutputDriver": "计算输出驱动电路。",
        "Precharger": "计算位线预充电电路。",
        "PredecodeBlock": "计算预译码电路。",
        "Result": "比较并输出 DESTINY 搜索结果。",
        "RowDecoder": "计算行译码电路。",
        "SenseAmp": "计算 DESTINY 原生感应放大器。",
        "SubArray": "计算子阵列及其读写路径。",
        "TSV": "提供仍被 DESTINY 引用的层间互连模型。",
        "Technology": "提供工艺节点与晶体管参数。",
        "Wire": "计算金属线寄生与驱动开销。",
        "formula": "提供电路模型共用的计算公式。",
        "core": "连接器件、缓存、搜索和结果汇总。",
        "m3d": "计算层间互连延迟、能量与台阶面积。",
        "bti": "保留旧版本 BTI 分析接口，v9 不调用刷新模型。",
        "edram": "计算 eDRAM 读出与 Si-eDRAM 刷新。",
        "sense_amp": "选择电压型或电流型读出电路。",
        "nonideal": "估算阵列非理想效应与误码。",
        "openvla_trace": "保留早期张量访问生成接口供兼容使用。",
        "v9_reports": "从仿真结果生成本地交付报告和 README。",
    }
    def describe(path):
        p = Path(path)
        if path.startswith("config/devices/"):
            return "提供 DESTINY 的器件参数。" if p.suffix == ".cell" else "指定该器件的 DESTINY 搜索范围。"
        if path.startswith("config/device_library"):
            return "保存器件指标和电路模型参数。"
        if path.startswith("config/scope_"):
            return "定义该版本的容量、工作负载和比较配置。"
        if p.stem in descriptions:
            return descriptions[p.stem] + ("头文件声明其接口。" if p.suffix == ".h" else "")
        if path.startswith("tests/"): return "检查模型公式、配置和功能是否正确。"
        if p.name.startswith("LICENSE"): return "保存适用代码的许可条款与版权。"
        return {
            ".gitignore": "排除生成文件和本地报告。", "README.md": "介绍项目、运行方式和实际示例结果。",
            "Makefile": "编译两个模拟器并提供运行入口。", "NOTICE": "说明第三方代码和许可归属。",
            "CONTRIBUTING": "说明如何提交改动和验证模型。",
            "main.cpp": "提供命令行程序入口。", "scope.py": "启动 Python 缓存评估器。",
            "__main__.py": "支持以 Python 模块方式启动。", "__init__.py": "导出公共模型接口。",
            "cli.py": "提供命令行入口。", "constant.h": "定义电路计算常量。",
            "global.h": "声明 DESTINY 共用对象。", "macros.h": "定义共用宏。",
            "typedef.h": "定义电路类型和枚举。", "scope-banner.png": "显示项目横幅。",
            "ci.yml": "在提交和拉取请求中自动构建并测试。",
            "bug_report.yml": "引导提交可复现的问题。",
            "CODEOWNERS": "指定默认代码维护者。",
            "PULL_REQUEST_TEMPLATE.txt": "提示说明变更、验证和数据来源。",
        }.get(p.name, "为模型运行提供辅助内容。")
    tree = {}
    for name in names + ["revision.md", "comparison.md", "file_structure.md"]:
        node = tree
        for part in Path(name).parts:
            node = node.setdefault(part, {})
    output = ["# 文件结构", "", "列出交付源码和三份本地报告；构建产物、缓存与原始仿真 JSON 单独列于末尾。", "", "```text", "SCOPE/"]
    def walk(node, prefix="", base=""):
        items = sorted(node.items())
        for i, (part, children) in enumerate(items):
            final = i == len(items)-1
            path = base + part
            label = part + ("/" if children else " — " + (
                {"revision.md": "展示本次关键公式和代码。",
                 "comparison.md": "汇总四种配置的实算结果。",
                 "file_structure.md": "解释交付文件的组织。"}.get(part) or describe(path)))
            output.append(prefix + ("└── " if final else "├── ") + label)
            if children: walk(children, prefix + ("    " if final else "│   "), path + "/")
    walk(tree)
    output += ["```", "", "本地生成目录：obj/ 为编译中间文件；config/.scope-cache/ 为 DESTINY 搜索缓存；results/ 为仿真 JSON；tmp/ 为临时资料。destiny 和 scope_model 是编译后的程序。"]
    (ROOT / "file_structure.md").write_text("\n".join(output) + "\n")

if __name__ == "__main__":
    main()
