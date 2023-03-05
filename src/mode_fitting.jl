using FastClosures
using LsqFit
using EchelleUtils

function refine_modes_centroid(spec, modes, spacing; n_iterations=3)
    modes_out = copy(modes)
    xarr = 1:length(spec)
    good = findall(isfinite.(spec))
    xrange = [good[1], good[end]]
    spec_rep = Maths.interp1d_badpix(xarr, spec, xrange)
    for _=1:n_iterations
        for j in eachindex(modes)
            use = findall((xarr .>= floor(modes[j] - spacing[j] / 2)) .&& (xarr .<= ceil(modes[j] + spacing[j] / 2)))
            xx, yy = @views xarr[use], spec_rep[use]
            modes_out[j] = Maths.weighted_mean(xx, yy)
        end
    end
    return modes_out
end

function fit_mode(spec, mode0, w, model=nothing; μ_bounds=[-1, 1], σ_bounds, background_poly_deg=0)

    # Window
    nx = length(spec)
    xarr = 1:nx
    use = findall((xarr .>= floor(mode0 - w / 2)) .&& (xarr .<= ceil(mode0 + w / 2)) .&& isfinite.(spec))
    xx, yy = xarr[use], spec[use]

    # The model
    if isnothing(model)
        model = @closure (x, pars) -> begin
            return Maths.gauss(x, pars[1], pars[2], pars[3]) .+ Polynomial(pars[4:end]).(x .- nanmean(x))
        end
    end

    # Remove approx baseline and normalize
    yy .-= nanminimum(yy)
    yy ./= nanmaximum(yy)

    # Initial parameters and bounds
    σ_guess = nanmean(σ_bounds)
    p0 = [1.0, mode0,               σ_guess,      0.1]
    lb = [0.7, mode0 + μ_bounds[1], σ_bounds[1], -0.5]
    ub = [1.3, mode0 + μ_bounds[2], σ_bounds[2],  0.5]

    if background_poly_deg > 0
        for i=1:background_poly_deg
            push!(p0, 0.01)
            push!(lb, -0.5)
            push!(ub, 0.5)
        end
    end

    # Fit
    result = LsqFit.curve_fit(model, xx, yy, p0, lower=lb, upper=ub)

    # Return
    return result
end