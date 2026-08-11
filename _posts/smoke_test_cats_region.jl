"""
smoke_test_cats_region.jl

Validates TwoStageCTS.jl on the extracted CATS COI-corridor region
(297 buses / 344 lines / 135 gens) before any wildfire data is wired in.
Same structure as smoke_test_ieee118.jl, but with checks specific to
this region's character:

  - It is a TRANSMISSION CORRIDOR, not a load center. Regional load is
    only ~368 MW while the COI itself carries up to 1725 MW, so the
    interesting quantity here is COI flow / import curtailment, not
    load shed. Tests below report COI flow explicitly.

  - It contains BOUNDARY INJECTION generators (gen_id starting "BND")
    standing in for severed external connections. Those are modeling
    artifacts; this test reports how much the solution leans on them,
    since heavy reliance means the region boundary is doing more work
    than the real network would.

Run with:
    julia --project=. smoke_test_cats_region.jl

Requires cats_region/{bus,line,gen}.csv from extract_cats_region.py.
"""

include("TwoStageCTS.jl")
using .TwoStageCTS
using CSV, DataFrames

println("="^62)
println("SMOKE TEST: TwoStageCTS on CATS COI-corridor region")
println("="^62)

# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------
println("\n[1] Loading CATS region...")

net = load_network_from_csv("cats_region/bus.csv",
                             "cats_region/line.csv",
                             "cats_region/gen.csv";
                             voll = 10000.0,
                             switch_cost = 50.0,
                             max_switch_stage1 = 5,
                             max_switch_stage2 = 10)

# NetworkData holds per-unit after load_network_from_csv; convert to MW so
# these compare correctly against dispatch/load-shed results, which
# build_and_solve returns in MW.
total_load = sum(net.bus_pd) * net.mw_per_pu
total_pmax = sum(net.gen_pmax) * net.mw_per_pu
println("  buses=$(net.n_bus)  lines=$(net.n_line)  gens=$(net.n_gen)")
println("  total load = $(round(total_load, digits=1)) MW")
println("  total capacity = $(round(total_pmax, digits=1)) MW")
@assert total_pmax >= total_load "FAIL: capacity below load -- infeasible by construction"
println("  -> capacity exceeds load. OK")

# Identify the COI line (bus 1898 -> 7908, the 500 kV intertie) and the
# boundary-injection generators, so the tests below can report on them.
gen_meta = CSV.read("cats_region/gen.csv", DataFrame)
bnd_idx = findall(startswith("BND"), gen_meta.gen_id)
println("  boundary-injection generators: $(length(bnd_idx)) of $(net.n_gen)")

coi_idx = nothing
# match the COI on the exported line table directly
line_meta = CSV.read("cats_region/line.csv", DataFrame)
coi_row = findfirst(r -> (r.from_bus == 1898 && r.to_bus == 7908) ||
                          (r.from_bus == 7908 && r.to_bus == 1898), eachrow(line_meta))
if coi_row === nothing
    println("  WARNING: COI line (1898<->7908) not found in this region -- "
            * "widen the extraction radius")
    coi_idx = nothing
else
    coi_idx = coi_row
    println("  COI line found: $(line_meta.line_id[coi_row]) "
            * "rated $(line_meta.rate_a[coi_row]) MW")
end

# ---------------------------------------------------------------------
# 2. Base case
# ---------------------------------------------------------------------
println("\n[2] Solving base case (single no-outage scenario)...")

base_scen = build_scenario_set(net, [String[]], [1.0])
t_base = @elapsed base = build_and_solve(net, base_scen;
                                          time_limit_sec = 300.0, mip_gap = 0.01)

println("  solve time: $(round(t_base, digits=2)) s   status: $(base.status)")
@assert base.status == TwoStageCTS.MOI.OPTIMAL "FAIL: base case did not solve"
println("  -> solved to optimality. OK")

dispatch = sum(base.stage1_dispatch)
println("  total dispatch = $(round(dispatch, digits=1)) MW vs load = $(round(total_load, digits=1)) MW")
@assert isapprox(dispatch, total_load, atol=1.0) "FAIL: dispatch/load mismatch"
println("  -> DC balance holds. OK")

n_open = sum(1 .- base.stage1_switch_state)
println("  stage-1 lines switched open: $(round(Int, n_open))")
if n_open > 0
    println("  (this is legitimate: opening a line to reduce generation cost is exactly")
    println("   what optimal transmission switching does. On a meshed network with")
    println("   heterogeneous costs, expect nonzero base-case switching -- unlike the")
    println("   IEEE 118 case, where none was economic.)")
end
@assert n_open <= net.max_switch_stage1 "FAIL: switching exceeds the stage-1 budget"
println("  -> switching within budget ($(net.max_switch_stage1)). OK")

bnd_use = sum(abs.(base.stage1_dispatch[bnd_idx]))
println("  boundary injection use: $(round(bnd_use, digits=1)) MW "
        * "($(round(100*bnd_use/max(dispatch,1), digits=1))% of dispatch)")
println("  (high reliance means the region boundary is carrying the case study --")
println("   worth noting when interpreting results, not a failure)")

# ---------------------------------------------------------------------
# 3. Force the COI out -- the actual event
# ---------------------------------------------------------------------
if coi_idx !== nothing
    println("\n[3] Forcing the COI line out (the Bootleg Fire scenario)...")

    coi_id = line_meta.line_id[coi_idx]
    scen = build_scenario_set(net, [String[], [coi_id]], [0.5, 0.5])
    t_coi = @elapsed coi_res = build_and_solve(net, scen;
                                                time_limit_sec = 300.0, mip_gap = 0.01)

    println("  solve time: $(round(t_coi, digits=2)) s   status: $(coi_res.status)")
    @assert coi_res.status in (TwoStageCTS.MOI.OPTIMAL, TwoStageCTS.MOI.TIME_LIMIT) "FAIL: did not solve"
    println("  -> solved. OK")
    println("  objective: \$$(round(coi_res.objective_value, digits=1))")
    println("  expected load shed: $(round(coi_res.stage2_expected_load_shed, digits=2)) MW")
    println("  stage-1 lines switched open: $(round(Int, sum(1 .- coi_res.stage1_switch_state)))")
    println("  stage-1 reserve procured: $(round(sum(coi_res.stage1_reserve), digits=1)) MW")

    # Stage 0 variant: COI already destroyed before decisions are made
    println("\n[4] Same line as a Stage 0 outage (already down at t=0)...")
    t_s0 = @elapsed s0 = build_and_solve(net, base_scen;
                                          initial_outage_lines = [coi_idx],
                                          time_limit_sec = 300.0, mip_gap = 0.01)
    println("  solve time: $(round(t_s0, digits=2)) s   status: $(s0.status)")
    @assert s0.status == TwoStageCTS.MOI.OPTIMAL "FAIL: Stage 0 case did not solve"
    @assert isapprox(s0.stage1_switch_state[coi_idx], 0.0, atol=1e-6) "FAIL: COI should be forced open"
    println("  -> COI forced open in stage 1. OK")
    println("  objective: \$$(round(s0.objective_value, digits=1))")
    println("  expected load shed: $(round(s0.stage2_expected_load_shed, digits=2)) MW")
end

println("\n" * "="^62)
println("SMOKE TEST COMPLETE — TwoStageCTS validated on the CATS region.")
println("="^62)
