"""
smoke_test_ieee118.jl

Validates TwoStageCTS.jl at real scale (118 buses, 186 lines, 54
generators) using the PGLib-derived bus.csv/line.csv/gen.csv. Two
parts:

  1. A trivial single no-outage scenario -- confirms the network loads
     correctly and the base dispatch is sane (no forced outages, so
     nothing interesting should happen: full load served, no switching,
     no load shed).
  2. A small set of scenarios with real forced line outages -- confirms
     the model still solves at this scale when recourse actually has to
     do something, and reports solve time so you have a first read on
     how this scales before scenario counts grow into the 20-100 range.

Run with:
    julia --project=. smoke_test_ieee118.jl

Requires bus.csv, line.csv, gen.csv (from export_pglib_case118.py) in
the same directory.
"""

include("TwoStageCTS.jl")
using .TwoStageCTS

println("="^60)
println("SMOKE TEST: TwoStageCTS on IEEE 118-bus (PGLib data)")
println("="^60)

# ---------------------------------------------------------------------
# 1. Load the network
# ---------------------------------------------------------------------
println("\n[1] Loading network from CSV...")

net = load_network_from_csv("bus.csv", "line.csv", "gen.csv";
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
println("  total generation capacity = $(round(total_pmax, digits=1)) MW")
@assert total_pmax >= total_load "FAIL: generation capacity below total load -- infeasible by construction"
println("  -> generation capacity exceeds load. OK")

# ---------------------------------------------------------------------
# 2. Base case: single no-outage scenario
# ---------------------------------------------------------------------
println("\n[2] Solving base case (single no-outage scenario)...")

base_scenarios = build_scenario_set(net, [String[]], [1.0])
t_base = @elapsed base_result = build_and_solve(net, base_scenarios;
                                                 time_limit_sec = 120.0,
                                                 mip_gap = 0.01)

println("  solve time: $(round(t_base, digits=2)) s")
println("  status: $(base_result.status)")
@assert base_result.status == TwoStageCTS.MOI.OPTIMAL "FAIL: base case did not solve to optimality"
println("  -> solved to optimality. OK")

total_dispatch = sum(base_result.stage1_dispatch)
println("  total dispatch = $(round(total_dispatch, digits=1)) MW vs total load = $(round(total_load, digits=1)) MW")
@assert isapprox(total_dispatch, total_load, atol=1.0) "FAIL: dispatch does not match load in the no-outage base case"
println("  -> dispatch matches load (DC balance holds, no losses modeled). OK")

n_switched_open = sum(1 .- base_result.stage1_switch_state)
println("  lines switched open in stage 1: $(round(Int, n_switched_open))")
@assert n_switched_open == 0 "FAIL: expected no switching with a single no-outage scenario"
println("  -> no unnecessary switching. OK")

expected_shed = base_result.stage2_expected_load_shed
println("  expected load shed: $(round(expected_shed, digits=3)) MW")
@assert isapprox(expected_shed, 0.0, atol=1.0) "FAIL: expected zero load shed with no forced outages"
println("  -> zero load shed as expected. OK")

# ---------------------------------------------------------------------
# 3. A small set of scenarios with real forced outages
# ---------------------------------------------------------------------
println("\n[3] Solving with 3 scenarios featuring real forced line outages...")

# Pick a few line IDs to force out -- L10, L50, L100 are arbitrary but
# spread across the network; swap these for whatever your actual
# wildfire-derived scenario set names once ScenarioGenerator.jl is
# feeding this instead of hand-picked IDs.
outage_scenarios = build_scenario_set(
    net,
    [String[], ["L10"], ["L50", "L100"]],
    [0.6, 0.25, 0.15],
)

t_outage = @elapsed outage_result = build_and_solve(net, outage_scenarios;
                                                      time_limit_sec = 120.0,
                                                      mip_gap = 0.01)

println("  solve time: $(round(t_outage, digits=2)) s")
println("  status: $(outage_result.status)")
@assert outage_result.status in (TwoStageCTS.MOI.OPTIMAL, TwoStageCTS.MOI.TIME_LIMIT) "FAIL: did not solve"
println("  -> solved (or hit time limit with a feasible incumbent). OK")

println("  objective value: \$$(round(outage_result.objective_value, digits=1))")
println("  expected stage-2 load shed: $(round(outage_result.stage2_expected_load_shed, digits=3)) MW")
n_switched = sum(1 .- outage_result.stage1_switch_state)
println("  lines switched open in stage 1: $(round(Int, n_switched))")
println("  stage-1 total reserve procured: $(round(sum(outage_result.stage1_reserve), digits=1)) MW")

println("\n" * "="^60)
println("SMOKE TEST COMPLETE — TwoStageCTS validated at IEEE 118-bus scale.")
println("Solve times: base case $(round(t_base, digits=2))s, "
        * "3-scenario case $(round(t_outage, digits=2))s")
println("="^60)
