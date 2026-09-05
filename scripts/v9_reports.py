"""Generate local v9 delivery reports and the README from simulated JSON results."""
from pathlib import Path
import json
import re
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

def update_readme_example(readme, example):
    pattern = r"(^```[^\n]*\n)Best cache configuration[^\n]*\n.*?(^```[ \t]*$)"
    updated, count = re.subn(pattern, lambda match: match[1] + example + match[2],
                             readme, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise ValueError("README must contain exactly one fenced Best cache configuration example")
    return updated

def snippet(path, marker, lines=10):
    source = (ROOT / path).read_text().splitlines()
    start = next(i for i, line in enumerate(source) if marker in line)
    return f"{path}:{start+1}\n\n```python\n" + "\n".join(source[start:start+lines]) + "\n```\n"

def main():
    reports = {name: json.loads((ROOT / f"results/scope_v9_{name}.json").read_text())
               for name in ("attention", "ffn")}
    if any(item.get("power_metric") != "system_average"
           for report in reports.values() for item in report["architecture_comparison"]):
        raise ValueError("Re-run v9 with power_metric=system_average before generating this report")
    summary, geometry, goals, activity = [], [], [], []
    for workload, report in reports.items():
        trace = report["case_reports"]["optimized"]["hit_rate_model"]["trace_metadata"]
        activity.append([workload, trace["power_activity_model"],
                         trace["sample_accesses"], trace["power_activity_transactions_per_operator"],
                         trace["policy_frequency_hz"], trace["memory_access_rate_per_s"]])
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
               "DESTINY proxy capacity", "Selected search target"], geometry) + "\n\n" +
        table(["Workload", "Power activity assumption (not full-trace activity)",
               "Replayed transactions", "Analytical power-budget transactions/operator",
               "Assumed operator Hz", "Power-budget requests/s"], activity) + "\n")
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
                f'\n- FoM (1/ns/mW): {a["fom_per_ns_mw"]:.9g} (Atten), {f["fom_per_ns_mw"]:.9g} (FFN)\n')
    if not same:
        example += "\nFFN configuration: " + ", ".join(
            f"L{i+1} {dev} {size(cap)}" for i, (dev, cap) in enumerate(zip(f["devices"], f["capacities_bytes"]))) + "\n"
    readme_path = ROOT / "README.md"
    readme_path.write_text(update_readme_example(readme_path.read_text(), example))
    revision = "# v9 修改\n\n"
    revision += "Orin：192 KiB L1；4 MiB SRAM 面积按 1:3 分配 L2/L3。AsyFET 沿用原 TFET 参数；OSFET refresh=none。\n\n"
    revision += "恢复最初 v9（c87e22d）的系统平均功耗：$P=\\lambda E+P_{static}+P_{refresh}$；FoM=$1/(LP)$。显示 Power（mW），不再用各配置自己的串行服务时间作功耗分母。保留回填、脏逐出等能量统计修正，因此不是恢复旧数值。调用频率沿用5 Hz，同一算子的四配置使用相同访存频率；该频率是活动假设，不是已验证的GPU吞吐率。单次操作的 E/T 仅作诊断，不参与默认排名。\n\n"
    revision += "本次按用户选择恢复旧活动量近似：power_activity_model=legacy_v9_analytical。Attention/FFN每算子分别按1,917,056/2,373,136次计算，5 Hz对应9,585,280/11,865,680次每秒。完整重放的50,089,216/90,329,028次仍用于命中率与每请求成本估计，不冒充旧活动量计数。最终Power是旧活动预算下的估计，不是完整tile算子以5 Hz运行的预测功耗；所有配置统一该假设。\n\n"
    revision += "旧公式（BF16、128 B事务，每事务64个元素）：Attention取Q=ceil(S/tile_m)，读元素数4H²+(3+2Q)SH、写元素数5SH；FFN读元素数3HI+2SH+2SI、写元素数2SI+SH。读写分别向上取整转换为事务，再乘调用频率。权重按一次计，不含完整GEMM重放中的重复请求。\n\n"
    revision += snippet("scope/power.py", '    if operator == "attention":', 14) + "\n"
    revision += snippet("scope/core.py", '    activity_count = int(raw["trace_cycle_accesses"])', 18) + "\n"
    revision += snippet("scope/core.py", '        dynamic_power_mw = access_rate', 4) + "\n"
    revision += snippet("scope/core.py", '        total_power_mw = dynamic_power_mw', 7) + "\n"
    revision += "电阻项为 $I^2Rt$，其中 $I$ 仅表示独立导通电流，默认 0，避免重复计算充放电电流。\n\n"
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
    revision += "验证：运行 python3 -m unittest discover -s tests -v；Attention/FFN 四配置和三开关对比。通过公式、独立地址循环、LRU 重放及开关断言，不把预定排名作为验收条件。L1 条件相同，其命中率也相同。\n\n"
    revision += "FoM 增幅按 FoM_SAO/FoM_reference−1 计算；comparison.md 同时列出相对 S-O-O 的 10% 目标、相对 S-S-S 的 80% 目标，以及固定功率或延迟时需要达到的另一指标。这是目标检查，不修改仿真结果。\n\n"
    revision += "来源：Orin technical brief（README 链接）；M3D-MDA Table 1：R=5.5 Ω、C=0.1 fF、pitch=0.2 μm。H/W 默认从等面积正方形 bank 推导，读写摆幅 0.1/1.2 V 为模型输入。\n\n"
    revision += "保留并公开：L3 0.15 延迟缩放和器件 standby 估算。移除默认跨帧 0.88 修正，改为完整算子分块地址重放，支持取模/XOR 映射。结果为行为级预测，不是实测。MIT 仅覆盖新增代码，第三方版权保留。\n\n"
    revision += "FFN核查：SSS 的1 MiB L2遇到128×4096×2 B=1 MiB的权重条带，再叠加输入，发生反复逐出。仅将tile_n从128改为64的诊断重放，L2条件命中由1.0649%变为45.4125%；该实验改变事务数，不直接与默认功耗混用。SOO的48 MiB L3装不下86 MiB的单个FFN权重矩阵，group_m=8使其跨组重复扫描；仅将L3改为96 MiB的诊断重放，L3条件命中由0.3456%变为64.3985%。只改group_m=19，片外到达率从7.3908%降至2.6293%，而L3条件命中仍仅1.9337%。这些是行为级假设的敏感性证据，不是GPU实测，也未覆盖默认参数。不能按L3>L2>L1强行修改命中率。\n\n"
    revision += "功耗与访存核查的详细解释见 access-audit.md。旧访问期间功率较低不代表更省能量：较长的等待/回填时间会扩大E/T分母，甚至令搜索偏向缓慢阵列。恢复系统功耗后按相同请求速率重新搜索阵列，不能只替换表格数字。\n\n"
    revision += snippet("scope/power.py", "def access_power_uw", 7) + "\n"
    revision += snippet("scope/power.py", "def access_power_mw", 2) + "\n"
    revision += snippet("scope/core.py", "        for path in self._path_probabilities()", 12) + "\n"
    (ROOT / "revision.md").write_text(revision)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    names = sorted({p for p in names if (ROOT / p).is_file()})
    descriptions = {
        "AccessTrace": "选择兼容的旧地址模型或完整分块算子地址。",
        "TiledTrace": "按矩阵分块顺序展开 Attention 和 FFN 的全局访存。",
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
        "power": "提供功率单位换算和原v9访存活动量估算。",
        "access_audit": "核对访问计数和能量口径，生成本地核查报告。",
        "plot_v9_comparison": "把四种配置的结果画成对比图。",
        "v9_sensitivity": "独立扫描参数，保留每种假设对应的结果。",
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
