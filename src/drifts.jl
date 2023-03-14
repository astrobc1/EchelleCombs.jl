export compute_drifts_ccf, cross_correlate_modes, compute_drifts_like_modes

function compute_drifts_ccf(spec0, spec; xrange=nothing, n_modes_ccf=5, min_mode_spacing, ccf_model=:gauss)

    # Number of pixels
    nx = length(spec0)

    # Resolve xrange
    xrange = resolve_xrange(spec, xrange)

    # Pixel grid
    xarr = [1:nx;]

    # Remove continuum and baseline
    background0 = estimate_background(spec0, min_mode_spacing)
    spec0_nobg = spec0 .- background0

    # Continuum
    continuum0 = estimate_continuum(spec0_nobg, min_mode_spacing)

    # Normalized spectrum
    spec0_norm = spec0_nobg ./ continuum0

    # Use normalized spectrum find peaks
    good = findall(spec0_norm .> 0.6)
    modes_pixels, _ = Maths.group_peaks(good, sep=min_mode_spacing)
    modes_pixels = modes_pixels[2:end-1]
    n_modes = length(modes_pixels)

    # Mode spacing
    peak_spacing_pixels_poly = @views Polynomials.fit(modes_pixels[2:end], diff(modes_pixels), 1)

    # Refine
    modes_pixels = refine_modes_centroid(spec0, modes_pixels, peak_spacing_pixels_poly.(modes_pixels); n_iterations=3)

    # Loop over pixels and perform CCF
    n_modes_ccf2 = Int(floor(n_modes_ccf / 2))
    drifts = fill(NaN, n_modes)
    for i=1:n_modes
        k1 = max(1, i - n_modes_ccf2)
        k2 = min(n_modes, i + n_modes_ccf2)
        xxi = Int(round(modes_pixels[k1] - peak_spacing_pixels_poly(modes_pixels[k1]) / 2))
        xxf = Int(round(modes_pixels[k2] + peak_spacing_pixels_poly(modes_pixels[k2]) / 2))
        window = [xxi, xxf]
        if ccf_model == :quad
            lags = -1:1
            ccf = cross_correlate_modes(spec0, spec, window, lags)
            pfit = Polynomials.fit(lags, ccf)
            drifts[i] = -0.5 * pfit.coeffs[2] / pfit.coeffs[3]
        elseif ccf_model == :gauss
            s2 = Int(round(peak_spacing_pixels_poly(modes_pixels[i]) / 2))
            lags = -s2:s2
            ccf = cross_correlate_modes(spec0, spec, window, lags)
            pbest = fit_ccf_gauss(lags, ccf)
            drifts[i] = pbest[2]
        end
    end

    # Return
    return modes_pixels, drifts
end

function fit_ccf_gauss(lags, ccf)
    ccf = ccf .- nanminimum(ccf)
    ccf ./= nanmaximum(ccf)
    p0 = [1.0, 0.001, 10.0, 0.001]
    lb = [0.8, -1, 0.0, -0.5]
    ub = [1.2, 1, 10, 0.5]
    model = (x, p) -> begin
        return Maths.gauss(x, p[1:3]...) .+ p[4]
    end
    result = LsqFit.curve_fit(model, lags, ccf, p0, lower=lb, upper=ub)
    pbest = result.param
    return pbest
end

function shift1d(x, s)
    nx = length(x)
    x_out = fill(NaN, nx)
    for i in eachindex(x_out)
        ii = i - s
        if 1 <= ii <= nx
            x_out[i] = x[ii]
        end
    end
    return x_out
end

function cross_correlate_modes(spec1, spec2, xrange, lags)
    nx = length(spec1)
    xarr = 1:nx
    good = findall(xarr .>= xrange[1] .&& xarr .<= xrange[2])
    x = xarr[good]
    y1 = spec1[good]
    y2 = spec2[good]
    y1 .-= nanminimum(y1)
    y2 .-= nanminimum(y2)
    y1 ./= nansum(y1)
    y2 ./= nansum(y2)
    nx = length(y1)
    n_lags = length(lags)
    ccf = fill(NaN, n_lags)
    vec_cross = fill(NaN, nx)
    weights = ones(nx)
    for i=1:n_lags
        weights .= 1
        vec_cross .= y1 .* shift1d(y2, lags[i])
        good = findall(isfinite.(vec_cross) .&& weights .> 0 .&& isfinite.(weights))
        ccf[i] = @views nansum(weights[good] .* vec_cross[good]) / sum(weights[good])
    end
    return ccf
end


function compute_drifts_like_modes(modes_pixels0, modes_pixels; thresh=1)
    pixels = Float64[]
    drifts = Float64[]
    for i in eachindex(modes_pixels)
        d = abs.(modes_pixels[i] .- modes_pixels0)
        k = Maths.nanargminimum(d)
        if d[k] < thresh
            push!(pixels, modes_pixels0[k])
            push!(drifts, modes_pixels[i] .- modes_pixels0[k])
        end
    end
    return pixels, drifts
end

function compute_drifts_like_modes(modes_pixels0, modes_pixels_err0, modes_pixels, modes_pixels_err; thresh=1)
    pixels = Float64[]
    drifts = Float64[]
    drifts_err = Float64[]
    for i in eachindex(modes_pixels)
        d = abs.(modes_pixels[i] .- modes_pixels0)
        k = Maths.nanargminimum(abs.(d))
        if abs(d[k]) < thresh
            push!(pixels, (modes_pixels[i] + modes_pixels0[k]) / 2)
            push!(drifts, d[k])
            push!(drifts_err, sqrt(modes_pixels_err[i]^2 + modes_pixels_err0[k]^2))
        end
    end
    return pixels, drifts, drifts_err
end