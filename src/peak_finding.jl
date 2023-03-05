function get_modes(spec, min_mode_spacing; σ_bounds=[0.2, 4.0], xrange=nothing, background_poly_deg=0)

    # Number of points
    nx = length(spec)

    # Pixel grid
    xarr = [1:nx;]

    # Use only good data
    xrange = resolve_xrange(spec, xrange)
    xi, xf = xrange[1], xrange[2]

    # Smooth and normalize spectrum
    spec_smooth = Maths.quantile_filter1d(spec, width=3)
    background = estimate_background(spec_smooth, min_mode_spacing)
    continuum = estimate_continuum(spec_smooth .- background, min_mode_spacing)
    spec_norm = (spec_smooth .- background) ./ continuum

    # Use normalized spectrum find peaks
    good = findall(spec_norm .> 0.6)
    modes_pixels, _ = Maths.group_peaks(good, sep=min_mode_spacing)

    # Ignore first and last mode
    modes_pixels = modes_pixels[2:end-1]
    n_modes = length(modes_pixels)

    # Mode spacing as a linear function in pixel space
    peak_spacing_pixels_poly = @views Polynomials.fit(modes_pixels[2:end], diff(modes_pixels), 1)
    min_mode_spacing_pixels = nanminimum(peak_spacing_pixels_poly.(xi:xf))
    
    # First iteratively refine based on centroid
    modes_pixels .= refine_modes_centroid(spec, modes_pixels, peak_spacing_pixels_poly.(modes_pixels); n_iterations=3)

    # Fit results
    amplitudes = fill(NaN, n_modes)
    modes_pixels_out = fill(NaN, n_modes)
    σs = fill(NaN, n_modes)
    background_polys = Vector{Polynomial}(undef, n_modes)
    rms = fill(NaN, n_modes)

    # Gaussian + Background model
    model = (x, pars) -> begin
        return Maths.gauss(x, pars[1], pars[2], pars[3]) .+ Polynomial(pars[4:end]).(x .- nanmean(x))
    end

    # Fit peaks
    for i=1:n_modes

        # Fit
        result = fit_mode(spec, modes_pixels[i], peak_spacing_pixels_poly(modes_pixels[i]), model; σ_bounds, background_poly_deg)

        # Store results
        pbest = result.param
        amplitudes[i] = pbest[1]
        modes_pixels_out[i] = pbest[2]
        σs[i] = pbest[3]
        background_polys[i] = Polynomial(pbest[4:end])
        rms[i] = sqrt(nansum(result.resid.^2) / length(result.resid))

    end

    # Return
    return modes_pixels_out, amplitudes, σs, background_polys, rms
end

function estimate_background(spec, min_mode_spacing; smooth_width=nothing)
    if isnothing(smooth_width)
        smooth_width = nextodd(Int(round(2.5 * min_mode_spacing)))
    else
        smooth_width =nextodd(smooth_width)
    end
    spec_smooth = Maths.quantile_filter1d(spec, width=3)
    background = Maths.quantile_filter1d(spec_smooth, width=smooth_width, p=0)
    background .= Maths.poly_filter1d(1:length(spec), background, width=smooth_width, deg=2)
    return background
end

function estimate_continuum(spec, min_mode_spacing; smooth_width=nothing)
    if isnothing(smooth_width)
        smooth_width = nextodd(Int(round(2.5 * min_mode_spacing)))
    else
        smooth_width =nextodd(smooth_width)
    end
    spec_smooth = Maths.quantile_filter1d(spec, width=3)
    continuum = Maths.quantile_filter1d(spec_smooth, width=smooth_width, p=0.99)
    continuum .= Maths.poly_filter1d(1:length(spec), continuum, width=smooth_width, deg=2)
    return continuum
end