module Etalons

using EchelleUtils
using EchelleSpectra

using NaNStatistics
using Polynomials
using LsqFit

export get_etalon_modes

SPEED_OF_LIGHT_MPS = 299792458.0

"""
    get_peak_spacing(λ, λi, λf, ν0, Δν; deg=1)
Returns a polynomial that describes the peak spacing at a given pixel.
"""
function get_etalon_modes(λ, spec; ℱ_guess, σ_guess=[0.2, 1.4, 3.0])

    # Generate theoretical peaks
    nx = length(λ)
    xarr = [1:nx;]
    good = findall(isfinite.(spec))
    xi, xf = minimum(good), maximum(good)

    # Peak spacing
    spec_smooth = Maths.quantile_filter1d(spec1d, width=3)
    background = estimate_background(spec_smooth, min_mode_spacing)
    background[bad] .= NaN

    continuum = estimate_continuum(spec_smooth .- background, min_mode_spacing)
    continuum[bad] .= NaN
    spec_norm = (spec_smooth .- background) ./ continuum
    good = findall(spec_norm .> 0.8)
    modes_pixels, _ = group_peaks(good, sep=min_mode_spacing)
    modes_pixels = modes_pixels[2:end-1]
    n_modes = length(modes_pixels)
    peak_spacing_pixels = Polynomials.fit(modes_pixels[2:end], diff(modes_pixels), 1)

    # First iteratively refine based on centroid just to get the window right
    for i=1:10
        for j=1:n_modes

            # Window
            use = findall((xarr .>= floor(modes_pixels[j] - peak_spacing(modes_pixels[j]) / 2)) .&& (xarr .<= ceil(modes_pixels[j] + peak_spacing(modes_pixels[j]) / 2)))
            xx, yy = @views xarr[use], spec[use]

            # Centroid
            modes_pixels[j] = Maths.weighted_mean(xx, yy)

        end
    end

    # Fit results
    amplitudes = fill(NaN, n_modes)
    σs = fill(NaN, n_modes)
    offsets = fill(NaN, n_modes)
    rms = fill(NaN, n_modes)

    # Fit peaks
    for i=1:n_modes

        # Window
        use = findall((xarr .>= floor(modes_pixels[i] - peak_spacing(modes_pixels[i]) / 2)) .&& (xarr .<= ceil(modes_pixels[i] + peak_spacing(modes_pixels[i]) / 2)))
        xx, yy = xarr[use], spec[use]

        # Remove approx baseline
        yy .-= nanminimum(yy)
        peak_val = nanmaximum(yy)

        # Pars and bounds
        p0 = [peak_val, modes_pixels[i], σ_guess[2], 0.1 * peak_val]
        lb = [0.7 * peak_val, modes_pixels[i] - 1, σ_guess[1], -0.5 * peak_val]
        ub = [1.3 * peak_val, modes_pixels[i] + 1, σ_guess[3], 0.5 * peak_val]

        # Model
        model = (_, pars) -> begin
            return Maths.gauss(xx, pars[1], pars[2], pars[3]) .+ pars[4]
        end

        # Fit
        try
            result = LsqFit.curve_fit(model, xx, yy, p0, lower=lb, upper=ub)
            pbest = result.param
            amplitudes[i] = pbest[1]
            modes_pixels[i] = pbest[2]
            σs[i] = pbest[3]
            offsets[i] = pbest[4]
            rms[i] = Maths.rmsloss(model(xx, pbest), yy)
        catch
            nothing
        end
    end

    # Return
    return modes_pixels, modes_λ, amplitudes, σs, offsets, redχ2s
end

function estimate_background(spec, min_mode_spacing; percentile=0.01, poly_deg=2)
    spec_smooth = Maths.quantile_filter1d(spec, 3)
    background = Maths.generalized_quantile_filter1d(spec_smooth, width=Int(round(2 * min_mode_spacing)), p=percentile)
    background .= Maths.poly_filter([1:length(spec);], background, width=3 * min_mode_spacing / 2, deg=poly_deg)
    return background
end

function estimate_continuum(spec, min_mode_spacing; percentile=0.99, poly_deg=2)
    continuum = Maths.generalized_quantile_filter1d(spec, width=Int(round(2 * min_mode_spacing)), p=percentile)
    continuum .= Maths.poly_filter([1:length(spec);], background, width=3 * min_mode_spacing / 2, poly_deg=2)
    return continuum
end

# function fabry_perot_model(λ, ℱ, ℓ, c0)
#     δ = 2π * ℓ / λ
#     return @. 1 / (1 + ℱ * sin(δ / 2)^2) + c0
# end

# function fit_etalon_FP(λ, spec; ℱ_guess, fsr_guess, xrange=nothing)
#     min_mode_spacing_guess = fsr_guess
#     λf, λi = λ[1], λ[end]
#     Δλ = λf - λi
#     Δx = length(spec) - 1
#     scale = Δx / Δλ
#     background = estimate_background(spec, fsr_guess * scale)
#     continuum = estimate_continuum(spec .- background, fsr_guess * scale)
#     spec_norm = (spec .- background) ./ continuum
# end

function get_peaks(spec, min_spacing; xrange=nothing)
    if isnothing(xrange)
        good = findall(isfinite.(spec))
        xrange = minimum(good), maximum(good)
    end
    xrange = Int.(xrange)
    xi, xf = xrange[1], xrange[2]
    λf, λi = λ[xi], λ[xf]
    Δλ = λf - λi
    Δx = length(spec) - 1
    scale = Δx / Δλ
    background = estimate_background(spec, min_spacing)
    continuum = estimate_continuum(spec .- background, min_spacing)
    spec_norm = (spec .- background) ./ continuum
    good = findall(spec_norm .> 0.8)
    modes_pixels, _ = Maths.group_peaks(good, sep=min_mode_spacing)
    modes_pixels = modes_pixels[2:end-1]
    n_modes = length(modes_pixels)
    return modes_pixels
end

# function fit_etalon_FP_single_peaks(λ, spec; ℱ_guess, fsr_guess, xrange=nothing)
#     #@infiltrate
#     min_mode_spacing_guess = fsr_guess
#     if isnothing(xrange)
#         good = findall(isfinite.(spec))
#         xrange = minimum(good), maximum(good)
#     end
#     xrange = Int.(xrange)
#     xi, xf = xrange[1], xrange[2]
#     λf, λi = λ[xi], λ[xf]
#     Δλ = λf - λi
#     Δx = length(spec) - 1
#     scale = Δx / Δλ
#     background = estimate_background(spec, fsr_guess * scale)
#     continuum = estimate_continuum(spec .- background, fsr_guess * scale)
#     spec_norm = (spec .- background) ./ continuum
#     good = findall(spec_norm .> 0.8)
#     modes_pixels, _ = Maths.group_peaks(good, sep=min_mode_spacing)
#     modes_pixels = modes_pixels[2:end-1]
#     n_modes = length(modes_pixels)
#     peak_spacing_pixels = Polynomials.fit(modes_pixels[2:end], diff(modes_pixels), 1)

#     amplitudes = fill(NaN, n_modes)
#     fwhms = fill(NaN, n_modes)
#     offsets = fill(NaN, n_modes)
#     modes_λs = fill(NaN, n_modes)

#     # Fit peaks
#     for i=1:n_modes

#         # Window
#         use = findall((xarr .>= floor(modes_pixels[i] - peak_spacing(modes_pixels[i]) / 2)) .&& (xarr .<= ceil(modes_pixels[i] + peak_spacing(modes_pixels[i]) / 2)))
#         xx, yy = λ[use], spec[use]

#         # Remove approx baseline
#         yy .-= nanminimum(yy)
#         peak_val = nanmaximum(yy)

#         # Pars and bounds (amp, mu, gamma)
#         p0 = [peak_val, modes_pixels[i],]
#         lb = [0.7 * peak_val, modes_pixels[i] - 1, σ_guess[1], -0.5 * peak_val]
#         ub = [1.3 * peak_val, modes_pixels[i] + 1, σ_guess[3], 0.5 * peak_val]

#         # Model
#         model = (_, pars) -> begin
#             return lorentz(xx, pars[1:3]...) .+ pars[4]
#         end

#         # Fit
#         try
#             result = LsqFit.curve_fit(model, xx, yy, p0, lower=lb, upper=ub)
#             pbest = result.param
#             amplitudes[i] = pbest[1]
#             modes_pixels[i] = pbest[2]
#             fwhms[i] = pbest[3]
#             offsets[i] = pbest[4]
#             k = Int(round(modes_pixels[i]))
#             modes_λs[i] = λ[k]
#         catch
#             nothing
#         end
#     end
# end


end
