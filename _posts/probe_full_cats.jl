"""
probe_full_cats.jl

Cheapest possible probe of whether TwoStageCTS is tractable on the FULL
CATS network (8,870 buses / 10,823 branches / 3,892 gens).

Runs three escalating cases and stops at the first failure, so you find
the wall without waiting on a case that was never going to finish:

  A. Base case, ONE no-outage scenario, switching DISABLED
     (max_switch_stage1 = 0). This is essentially a DC OPF with the MILP
     machinery attached -- if this is slow, the problem is the network's
     numerical conditioning, not the combinatorics.

  B. Base case, ONE scenario, switching ENABLED (budget 5).
     Adds 10,823 binaries. The difference between A and B isolates the
     cost of the switching decision itself.

  C. THREE scenarios with a forced outage.
     Adds the stochastic structure. Expect this to be the hard one.

Each case has its own time limit; a TIME_LIMIT result is informative,
not a failure -- it tells you decomposition is required.

NOTE ON CONDITIONING: full CATS reactances span 1.21e-06 to 8.42e-01 pu,
so the DC flow equations carry matrix coefficients up to ~8.3e7. Solvers
generally want coefficient ranges under ~1e6. Watch the "Matrix" range
HiGHS prints -- if solve quality is poor, per-unit rescaling of the flow
variables (rather than MW) is the next lever, not more solver time.

Run with:
    julia --project=. probe_full_cats.jl
"""

include("TwoStageCTS.jl")
using .TwoStageCTS
using CSV, DataFrames

println("="^66)
println("TRACTABILITY PROBE: TwoStageCTS on FULL CATS")
println("="^66)

println("\nLoading full CATS...")
t_load = @elapsed net = load_network_from_csv(
    "full_cats/bus.csv", "full_cats/line.csv", "full_cats/gen.csv";
    voll = 10000.0, switch_cost = 50.0,
    max_switch_stage1 = 5, max_switch_stage2 = 10)

println("  loaded in $(round(t_load, digits=1)) s")
println("  buses=$(net.n_bus)  lines=$(net.n_line)  gens=$(net.n_gen)")
println("  total load = $(round(sum(net.bus_pd)*net.mw_per_pu, digits=0)) MW")
println("  total capacity = $(round(sum(net.gen_pmax)*net.mw_per_pu, digits=0)) MW")

b = net.line_susceptance
println("  susceptance range: $(round(minimum(b), digits=1)) to $(round(maximum(b), digits=1))")
println("  implied max matrix coefficient: $(round(net.base_mva * maximum(b), sigdigits=3))")

# Tunable from the command line, e.g.
#   julia --project=. probe_full_cats.jl --gap 0.02 --time-limit 900
mip_gap = 0.01
time_mult = 1.0
for (i, a) in enumerate(ARGS)
    # `global` is required: assigning inside a for-loop at top level
    # creates a new LOCAL in Julia's soft scope, silently leaving the
    # global unchanged (which is why --gap appeared to do nothing).
    a == "--gap" && (global mip_gap = parse(Float64, ARGS[i+1]))
    a == "--time-limit" && (global time_mult = parse(Float64, ARGS[i+1]) / 900.0)
end
println("\n  MIP gap tolerance: $(100*mip_gap)%")

results = NamedTuple[]

function probe(label, scenarios, net_used; limit)
    println("\n" * "-"^66)
    println(label)
    println("-"^66)
    t = @elapsed r = build_and_solve(net_used, scenarios;
                                      time_limit_sec = limit * time_mult,
                                      mip_gap = mip_gap)
    # A time limit WITH a feasible incumbent is not a failure: for a
    # receding-horizon application a good solution inside the decision
    # window matters more than a certificate of optimality. Only treat
    # "no feasible solution at all" as blocking.
    solved = r.objective_value !== missing
    println("  time: $(round(t, digits=1)) s   status: $(r.status)")
    if r.status == TwoStageCTS.MOI.TIME_LIMIT && r.objective_value !== missing
        println("  (hit the time limit with a feasible incumbent -- near-optimal,")
        println("   not unsolved. Raise --gap if you only need operational quality.)")
    end
    if r.objective_value !== missing
        println("  objective: \$$(round(r.objective_value, digits=1))")
        println("  expected load shed: $(round(r.stage2_expected_load_shed, digits=1)) MW")
        println("  lines open: $(round(Int, sum(1 .- r.stage1_switch_state)))")
    else
        println("  NO FEASIBLE SOLUTION FOUND within the time limit")
    end
    push!(results, (label=label, time=t, status=r.status, solved=solved))
    return r
end

base_scen = build_scenario_set(net, [String[]], [1.0])

# --- Case A: no switching allowed ---
net_noswitch = TwoStageCTS.NetworkData(
    net.n_bus, net.n_line, net.n_gen, net.line_from, net.line_to,
    net.line_susceptance, net.line_rate, net.line_id, net.switchable,
    net.gen_bus, net.gen_pmin, net.gen_pmax, net.gen_cost,
    net.gen_reserve_cost, net.bus_pd, net.ref_bus, net.voll,
    net.switch_cost, 0, 0, net.theta_max, net.base_mva, net.mw_per_pu)

rA = probe("CASE A: 1 scenario, switching DISABLED (essentially a DC OPF)",
           base_scen, net_noswitch; limit = 600.0)

if !results[end].solved
    println("\n" * "="^66)
    println("STOPPING: even with no switching decision, this did not solve.")
    println("That points at the network's numerical conditioning, not the")
    println("combinatorics -- rescaling to per-unit flows is the next step,")
    println("and decomposition would NOT fix it.")
    println("="^66)
else
    rB = probe("CASE B: 1 scenario, switching ENABLED (budget 5)",
               base_scen, net; limit = 900.0)

    if !results[end].solved
        println("\n" * "="^66)
        println("STOPPING: the switching binaries are the wall.")
        println("Case A solved, Case B did not -- so the network is")
        println("numerically fine and the combinatorics are the problem.")
        println("Scenario-wise decomposition (Benders/Lagrangian) is the")
        println("right next move.")
        println("="^66)
    else
        line_meta = CSV.read("full_cats/line.csv", DataFrame)
        coi = findfirst(r -> (r.from_bus == 1898 && r.to_bus == 7908) ||
                              (r.from_bus == 7908 && r.to_bus == 1898),
                        eachrow(line_meta))
        outage_id = coi === nothing ? line_meta.line_id[1] : line_meta.line_id[coi]
        if coi !== nothing
            println("\n(using the COI line $(outage_id) as the forced outage)")
        end

        scen3 = build_scenario_set(net, [String[], [outage_id], [outage_id]],
                                   [0.5, 0.3, 0.2])
        probe("CASE C: 3 scenarios with a forced outage", scen3, net; limit = 1800.0)
    end
end

println("\n" * "="^66)
println("PROBE SUMMARY")
println("="^66)
for r in results
    mark = r.solved ? "solved" : "DID NOT SOLVE"
    println("  $(rpad(round(r.time, digits=1), 8))s  $mark   $(r.label)")
end
println("="^66)
