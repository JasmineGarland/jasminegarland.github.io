"""
fire_to_network.jl

First real coupling of the fire and network sides: takes the wind-joined
GOES fire states and full CATS corridor geometry, and computes a
per-line exposure score at each timestep.

This is the checkable step. If the Bootleg Fire genuinely threatened the
California-Oregon Intertie, an exposure model built from independent
data (GOES detections + HRRR wind + CATS geometry, none of them fitted
to each other) should say so without being told. That is a validation of
the pipeline, not a tuned result.

Exposure follows ExposureProbability.jl's structure -- distance to the
fire, alignment of the corridor with the wind -- but is computed here
directly against real corridor polylines rather than the two-endpoint
placeholder geometry. Vegetation and terrain terms are omitted for now
(no raster join yet); they belong in a later revision, and their absence
should make this a CONSERVATIVE estimate of exposure rather than an
inflated one.

Outputs:
  exposure_timeseries.csv   line_id, timestamp, min_distance_km, wind_align, exposure
  exposure_summary.csv      per-line peak exposure and when it occurred

Run with:
    julia --project=. fire_to_network.jl \\
        --fire bootleg_jul9_13_wind.json \\
        --geometry full_cats/line_geometry.json

Needs: JSON, CSV, DataFrames, ProgressMeter (optional)
"""

using JSON
using CSV
using DataFrames
using Dates
using LinearAlgebra

const R_EARTH_KM = 6371.0

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

haversine_km(lon1, lat1, lon2, lat2) = begin
    p1, p2 = deg2rad(lat1), deg2rad(lat2)
    dp, dl = deg2rad(lat2 - lat1), deg2rad(lon2 - lon1)
    a = sin(dp / 2)^2 + cos(p1) * cos(p2) * sin(dl / 2)^2
    2 * R_EARTH_KM * asin(sqrt(min(1.0, a)))
end

"""
    point_seg_dist_km(p, a, b)

Distance from point `p` to the segment `a`-`b`, all as (lon, lat).

Uses a local equirectangular projection about the segment midpoint --
accurate to well under a kilometre at these latitudes and over these
distances, and far cheaper than a full geodesic segment distance for
the ~4 million point-segment tests this script performs.
"""
function point_seg_dist_km(p, a, b)
    lat0 = deg2rad((a[2] + b[2]) / 2)
    kx = R_EARTH_KM * cos(lat0) * pi / 180
    ky = R_EARTH_KM * pi / 180
    ax, ay = a[1] * kx, a[2] * ky
    bx, by = b[1] * kx, b[2] * ky
    px, py = p[1] * kx, p[2] * ky
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = L2 == 0 ? 0.0 : clamp(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    hypot(px - (ax + t * dx), py - (ay + t * dy))
end

"""
    line_bearing_deg(coords)

Dominant compass bearing of a corridor, from first to last vertex.
"""
function line_bearing_deg(coords)
    lon1, lat1 = deg2rad(coords[1][1]), deg2rad(coords[1][2])
    lon2, lat2 = deg2rad(coords[end][1]), deg2rad(coords[end][2])
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    mod(rad2deg(atan(y, x)) + 360, 360)
end

"""
    wind_alignment(bearing_deg, wind_from_deg)

0-1: 1 when the corridor runs parallel to the wind (fire runs along it),
0 when perpendicular. Wind direction is meteorological (FROM).
"""
function wind_alignment(bearing_deg, wind_from_deg)
    d = abs(mod(bearing_deg - wind_from_deg + 180, 360) - 180)
    abs(cos(deg2rad(d)))
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

function main()
    fire_path = "bootleg_jul9_13_wind.json"
    geom_path = "full_cats/line_geometry.json"
    max_dist_km = 25.0     # beyond this, exposure is treated as zero
    for (i, a) in enumerate(ARGS)
        a == "--fire" && (fire_path = ARGS[i+1])
        a == "--geometry" && (geom_path = ARGS[i+1])
        a == "--max-dist-km" && (max_dist_km = parse(Float64, ARGS[i+1]))
    end

    println("="^64)
    println("FIRE -> NETWORK COUPLING")
    println("="^64)

    fire = JSON.parsefile(fire_path)
    geom = JSON.parsefile(geom_path)
    states = fire["states"]
    lines = geom["lines"]
    println("  fire states: $(length(states))  ($(count(s -> s["n_pixels"] > 0, states)) with detections)")
    println("  CATS lines:  $(length(lines))")

    # Precompute per-line geometry once: coordinate list, bearing, bbox.
    # NOTE: 7 of 10,823 CATS lines carry MultiLineString-style nested
    # coordinates rather than a flat list of [lon, lat] pairs. Flatten
    # them here -- otherwise those lines throw a type error partway
    # through a long run.
    function flatten_coords(raw)
        out = Tuple{Float64,Float64}[]
        for v in raw
            if v[1] isa AbstractVector          # nested ring
                for w in v
                    push!(out, (Float64(w[1]), Float64(w[2])))
                end
            else
                push!(out, (Float64(v[1]), Float64(v[2])))
            end
        end
        return out
    end

    coords = [flatten_coords(l["coords"]) for l in lines]
    bearings = [length(c) >= 2 ? line_bearing_deg(c) : 0.0 for c in coords]
    ids = [l["line_id"] === nothing ? "UNMATCHED_$(l["cats_id"])" : String(l["line_id"]) for l in lines]
    kv = [l["kV"] === nothing ? missing : Float64(l["kV"]) for l in lines]

    lon_min = [minimum(p[1] for p in c) for c in coords]
    lon_max = [maximum(p[1] for p in c) for c in coords]
    lat_min = [minimum(p[2] for p in c) for c in coords]
    lat_max = [maximum(p[2] for p in c) for c in coords]

    rows = NamedTuple[]
    nstates = length(states)

    for (si, st) in enumerate(states)
        st["n_pixels"] == 0 && continue
        det = st["detections"]
        isempty(det) && continue
        pts = [(Float64(d[1]), Float64(d[2])) for d in det]

        wdir = st["wind_direction"]
        wspd = st["wind_speed"]

        # Fire bounding box, padded by max_dist_km, to skip the vast
        # majority of CATS lines cheaply. Without this the full
        # cross-product is ~4e9 point-segment tests.
        flon_min = minimum(p[1] for p in pts); flon_max = maximum(p[1] for p in pts)
        flat_min = minimum(p[2] for p in pts); flat_max = maximum(p[2] for p in pts)
        dlat = max_dist_km / 111.0
        dlon = max_dist_km / (111.0 * cos(deg2rad((flat_min + flat_max) / 2)))

        for li in eachindex(coords)
            (lon_max[li] < flon_min - dlon || lon_min[li] > flon_max + dlon) && continue
            (lat_max[li] < flat_min - dlat || lat_min[li] > flat_max + dlat) && continue

            c = coords[li]
            best = Inf
            for p in pts
                for k in 1:(length(c) - 1)
                    d = point_seg_dist_km(p, c[k], c[k+1])
                    d < best && (best = d)
                    best < 0.05 && break
                end
                best < 0.05 && break
            end
            best > max_dist_km && continue

            align = wdir === nothing ? missing : wind_alignment(bearings[li], Float64(wdir))
            dist_score = clamp(1.0 - best / max_dist_km, 0.0, 1.0)
            # Distance dominates; wind alignment modulates. Vegetation and
            # terrain terms are absent (see module docstring), so weights
            # are renormalised over the two available factors rather than
            # silently treating the missing ones as zero.
            expo = align === missing ? dist_score : 0.65 * dist_score + 0.35 * align

            push!(rows, (line_id = ids[li], kV = kv[li],
                         timestamp = st["timestamp"],
                         min_distance_km = round(best, digits = 3),
                         wind_align = align === missing ? missing : round(align, digits = 3),
                         wind_speed = wspd,
                         exposure = round(expo, digits = 4)))
        end

        if si % 25 == 0 || si == nstates
            println("  [$si/$nstates] $(st["timestamp"])  cumulative line-timesteps: $(length(rows))")
        end
    end

    if isempty(rows)
        println("\nNo CATS line came within $(max_dist_km) km of any detection.")
        println("Either the fire is outside the network footprint, or the")
        println("geometry files are misaligned -- check before proceeding.")
        return
    end

    df = DataFrame(rows)
    CSV.write("exposure_timeseries.csv", df)

    summary = combine(groupby(df, :line_id),
                      :exposure => maximum => :peak_exposure,
                      :min_distance_km => minimum => :closest_km,
                      :kV => first => :kV,
                      nrow => :n_timesteps)
    sort!(summary, :peak_exposure, rev = true)
    CSV.write("exposure_summary.csv", summary)

    println("\n" * "="^64)
    println("RESULTS")
    println("="^64)
    println("  distinct lines exposed: $(nrow(summary))")
    println("  line-timestep records:  $(nrow(df))")
    println("\n  most exposed corridors:")
    println("  ", rpad("line_id", 12), rpad("kV", 8), rpad("closest_km", 12), "peak_exposure")
    for r in eachrow(first(summary, 15))
        println("  ", rpad(r.line_id, 12),
                rpad(r.kV === missing ? "?" : string(Int(r.kV)), 8),
                rpad(round(r.closest_km, digits = 2), 12),
                round(r.peak_exposure, digits = 3))
    end

    # The specific check: does the COI appear? Bus 1898 <-> 7908 is the
    # 500 kV intertie the Bootleg Fire actually affected.
    coi = filter(l -> (l["f_bus"] == 1898 && l["t_bus"] == 7908) ||
                      (l["f_bus"] == 7908 && l["t_bus"] == 1898), lines)
    if !isempty(coi)
        coi_id = coi[1]["line_id"]
        hit = filter(r -> r.line_id == String(coi_id), summary)
        println("\n  COI line ($(coi_id)):")
        if nrow(hit) > 0
            println("    EXPOSED -- closest $(round(hit.closest_km[1], digits=2)) km, "
                    * "peak exposure $(round(hit.peak_exposure[1], digits=3))")
            println("    The pipeline independently identifies the corridor that was")
            println("    actually affected. That is the coherence check.")
        else
            println("    not exposed within $(max_dist_km) km.")
            println("    Worth investigating before drawing conclusions: the fire burned")
            println("    in Oregon, north of the CATS footprint, so the corridor may lie")
            println("    outside the modelled network even though the real line was hit.")
        end
    end

    println("\n  wrote exposure_timeseries.csv and exposure_summary.csv")
    println("="^64)
end

main()
