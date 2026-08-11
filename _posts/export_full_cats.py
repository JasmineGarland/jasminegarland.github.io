"""
export_full_cats.py

Export the ENTIRE CATS California Test System (8,870 buses / 10,823
branches / 3,892 gens) to the formats the pipeline consumes:

    full_cats/bus.csv            for TwoStageCTS.load_network_from_csv
    full_cats/line.csv
    full_cats/gen.csv
    full_cats/cats.m             for PowerModels (AC feasibility)
    full_cats/line_geometry.json for ExposureProbability

This replaces extract_cats_region.py. No geographic filtering, no
connectivity pruning, no boundary-injection generators -- all of which
were artifacts of subsetting. CATS already models California's external
interties itself (e.g. bus 1898, 500 kV, tagged IMPORT, at the
Malin/Captain Jack area), so inheriting the authors' treatment is both
simpler and more defensible than inventing boundary conditions no
comparable paper uses.

Known approximation (unchanged from the region export): CATS gencost is
quadratic (c2*P^2 + c1*P + c0) with nonzero c2 for many units, and
TwoStageCTS supports only linear costs, so only c1 is exported. Revisit
with PWL segments before trusting dispatch economics for publication.

Usage:
    python3 export_full_cats.py --repo cats_repo --outdir full_cats
"""

import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd


def parse_matpower(path):
    """Pull bus/gen/gencost/branch matrices out of a MATPOWER .m file."""
    with open(path) as f:
        text = f.read()

    def block(name):
        m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
        if not m:
            raise ValueError(f"no mpc.{name} block in {path}")
        rows = []
        for line in m.group(1).strip().split("\n"):
            line = line.split("%")[0].strip().rstrip(";").strip()
            if line:
                rows.append([float(v) for v in line.split()])
        return rows

    base = float(re.search(r"mpc\.baseMVA\s*=\s*([0-9.]+)", text).group(1))
    return dict(baseMVA=base, bus=block("bus"), gen=block("gen"),
                gencost=block("gencost"), branch=block("branch"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="cats_repo")
    ap.add_argument("--outdir", default="full_cats")
    args = ap.parse_args()

    mp_path = os.path.join(args.repo, "MATPOWER", "CaliforniaTestSystem.m")
    gis_line = os.path.join(args.repo, "GIS", "CATS_lines.json")

    print("Parsing full CATS MATPOWER file...")
    mpc = parse_matpower(mp_path)
    print(f"  {len(mpc['bus'])} buses, {len(mpc['branch'])} branches, "
          f"{len(mpc['gen'])} gens, baseMVA={mpc['baseMVA']}")

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- bus.csv ----------------
    # MATPOWER bus cols: bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    bus_ids = [int(r[0]) for r in mpc["bus"]]
    types = [int(r[1]) for r in mpc["bus"]]
    n_ref = sum(1 for t in types if t == 3)
    if n_ref != 1:
        print(f"  NOTE: found {n_ref} slack buses (type 3); "
              f"keeping the first as reference")
    ref_seen = False
    is_ref = []
    for t in types:
        if t == 3 and not ref_seen:
            is_ref.append(1)
            ref_seen = True
        else:
            is_ref.append(0)

    bus_df = pd.DataFrame({
        "bus_id": bus_ids,
        "pd": [r[2] for r in mpc["bus"]],
        "is_ref": is_ref,
    })
    bus_df.to_csv(os.path.join(args.outdir, "bus.csv"), index=False)
    print(f"  bus.csv   {len(bus_df):6d} buses  (load {bus_df.pd.sum():.0f} MW, "
          f"ref bus {bus_df.loc[bus_df.is_ref == 1, 'bus_id'].iloc[0]})")

    # ---------------- line.csv ----------------
    # branch cols: fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
    line_df = pd.DataFrame({
        "line_id": [f"L{i}" for i in range(len(mpc["branch"]))],
        "from_bus": [int(r[0]) for r in mpc["branch"]],
        "to_bus": [int(r[1]) for r in mpc["branch"]],
        "x": [r[3] for r in mpc["branch"]],
        "rate_a": [r[5] for r in mpc["branch"]],
    })
    n_zero = int((line_df.rate_a == 0).sum())
    line_df.to_csv(os.path.join(args.outdir, "line.csv"), index=False)
    print(f"  line.csv  {len(line_df):6d} lines  "
          f"(x {line_df.x.min():.2e}-{line_df.x.max():.2e} pu, "
          f"{n_zero} unrated)")

    # ---------------- gen.csv ----------------
    gen_rows = []
    n_quad = 0
    for i, r in enumerate(mpc["gen"]):
        c = mpc["gencost"][i] if i < len(mpc["gencost"]) else None
        if c and len(c) >= 7:
            c2, c1 = c[4], c[5]
        else:
            c2, c1 = 0.0, 50.0
        if c2 != 0.0:
            n_quad += 1
        gen_rows.append(dict(gen_id=f"G{i}", bus=int(r[0]),
                             pmin=r[9], pmax=r[8],
                             cost=c1, reserve_cost=0.10 * c1))
    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(os.path.join(args.outdir, "gen.csv"), index=False)
    print(f"  gen.csv   {len(gen_df):6d} gens   "
          f"(capacity {gen_df.pmax.sum():.0f} MW, "
          f"cost {gen_df.cost.min():.1f}-{gen_df.cost.max():.1f} $/MWh)")
    if n_quad:
        print(f"            NOTE: {n_quad} gens have nonzero quadratic cost; "
              f"only the linear term exported")
    n_negmin = int((gen_df.pmin < 0).sum())
    if n_negmin:
        print(f"            NOTE: {n_negmin} gens have negative pmin "
              f"(can absorb power) -- with positive $/MWh this REDUCES the "
              f"objective. Watch for negative objectives.")

    # ---------------- cats.m (AC, verbatim copy) ----------------
    # No renumbering needed: we keep every bus, so CATS's own ids stand.
    with open(mp_path) as src, \
         open(os.path.join(args.outdir, "cats.m"), "w") as dst:
        dst.write(src.read())
    print(f"  cats.m    full MATPOWER case copied verbatim (AC parameters intact)")

    # ---------------- line_geometry.json ----------------
    # Matching geometry to branches needs care: CATS contains PARALLEL
    # CIRCUITS -- multiple distinct branches between the same bus pair.
    # Keying a dict on (from_bus, to_bus) collapses them, which silently
    # assigns several branches the same line_id and leaves others with no
    # geometry at all (249 lines, 232 duplicated ids when we did that).
    # Instead, consume branch indices from a per-bus-pair queue so each
    # geometry feature claims a distinct branch.
    with open(gis_line) as f:
        lines_gis = json.load(f)

    from collections import deque
    pair_queue = defaultdict(deque)
    for i, r in enumerate(mpc["branch"]):
        pair_queue[(int(r[0]), int(r[1]))].append(i)

    geo = []
    unmatched = 0
    for feat in lines_gis["features"]:
        p = feat["properties"]
        key = (int(p["f_bus"]), int(p["t_bus"]))
        if pair_queue[key]:
            idx = pair_queue[key].popleft()
            lid = f"L{idx}"
        else:
            lid = None
            unmatched += 1
        geo.append(dict(
            line_id=lid,
            cats_id=p["CATS_ID"], f_bus=key[0], t_bus=key[1],
            kV=p["kV"], transformer=bool(p["transformer"]),
            coords=feat["geometry"]["coordinates"],
        ))

    matched = sum(1 for g in geo if g["line_id"] is not None)
    dup = len(geo) - len({g["line_id"] for g in geo if g["line_id"]}) - unmatched
    with open(os.path.join(args.outdir, "line_geometry.json"), "w") as f:
        json.dump(dict(n_lines=len(geo), lines=geo), f)
    print(f"  line_geometry.json  {len(geo)} features "
          f"({matched} matched to a distinct branch, {unmatched} unmatched, "
          f"{dup} duplicate ids)")

    print(f"\nDone. Full CATS exported to {args.outdir}/")
    print(f"  Scale check: {len(line_df)} switching binaries PER SCENARIO.")
    print(f"  For reference, IEEE 118 had 186 and solved 20 scenarios in ~1.3 s.")


if __name__ == "__main__":
    main()
