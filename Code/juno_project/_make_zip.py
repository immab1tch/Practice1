# -*- coding: utf-8 -*-
"""打包 小组交付_v61_20261026.zip (Prompt 07 §五)"""
import zipfile, os

root = r"C:\ModelingProgram\Practice1"
jp = os.path.join(root, "Code", "juno_project")
fu = os.path.join(jp, "results", "figures_upgrade")
out = os.path.join(root, "小组交付_v61_20261026.zip")

files = [
    os.path.join(fu, "fig02_architecture.png"),
    os.path.join(fu, "fig04_pareto.png"),
    os.path.join(fu, "fig08_m5_flow.png"),
    os.path.join(jp, "results", "m3", "m3_elasticity.csv"),
    os.path.join(jp, "results", "m3", "m3_scenarios.csv"),
    os.path.join(jp, "results", "m3", "tau_lever_v61.csv"),
    os.path.join(jp, "results", "m1", "run_log.txt"),
    os.path.join(jp, "results", "m2", "run_log.txt"),
    os.path.join(jp, "results", "m5", "run_log.txt"),
    os.path.join(jp, "results", "m3", "run_log.txt"),
    os.path.join(jp, "results", "m5", "m5_region_allocation_v61.csv"),
]

# zip 内名称: run_log 用模块前缀区分
arcnames = {
    os.path.join(jp, "results", "m1", "run_log.txt"): "m1_run_log.txt",
    os.path.join(jp, "results", "m2", "run_log.txt"): "m2_run_log.txt",
    os.path.join(jp, "results", "m5", "run_log.txt"): "m5_run_log.txt",
    os.path.join(jp, "results", "m3", "run_log.txt"): "m3_run_log.txt",
}

missing = [f for f in files if not os.path.isfile(f)]
if missing:
    print("MISSING:", *missing, sep="\n")
    raise SystemExit(1)

if os.path.exists(out):
    os.remove(out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, arcnames.get(f, os.path.basename(f)))
print("ZIP OK:", out)
for f in files:
    print("  +", arcnames.get(f, os.path.basename(f)), os.path.getsize(f), "B")
