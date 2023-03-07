using PyCall

export fit_peaks_cheb2d, get_chebyvals, build_λsolution_chebyval2d, build_λsolution_chebyval2d_flat

## 2d CC polynomial
function build_λsolution_chebyval2d(pixels, orders, max_pixel, max_order, coeffs)
    nx = length(pixels)
    nm = length(orders)
    m, n = size(coeffs)
    λ = fill(NaN, (nm, nx))
    for i=1:nx
        for o=1:nm
            s = 0.0
            for j=1:n
                for k=1:m
                    s += coeffs[k, j] * chebyval(pixels[i] / max_pixel, j-1) * chebyval(orders[o] / max_order, k-1) / orders[o]
                end
            end
            λ[o, i] = s
        end
    end
    return λ
end

function build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, coeffs, orders)
    nx = length(chebs_pixels)
    λ = zeros(nx)
    m, n = size(coeffs)
    for i=1:nx
        s = 0.0
        for j=1:n
            for k=1:m
                s += coeffs[k, j] * chebs_pixels[i][j] * chebs_orders[i][k] / orders[i]
            end
        end
        λ[i] = s
    end
    return λ
end

function fit_peaks_cheb2d(pixel_centers, orders, λ_centers, weights, max_pixel, max_order, nx, deg_inter_order, deg_intra_order, n_iterations=1, max_vel_cut=200)

    # Initial params and weights
    p0 = ones((deg_inter_order + 1) * (deg_intra_order + 1)) / 100
    pixel_centers_running = copy(pixel_centers)
    λ_centers_running = copy(λ_centers)
    weights_running = copy(weights)
    coeffs_best = copy(p0)

    # Load scipy
    scipyopt = pyimport("scipy.optimize")

    for i=1:n_iterations

        # Update bad weights
        bad = findall(.~isfinite.(weights_running) .|| (weights_running .== 0) .|| .~isfinite.(pixel_centers) .|| .~isfinite.(λ_centers))
        weights_running[bad] .= 0
        pixel_centers_running[bad] .= 0
        λ_centers_running[bad] .= 0

        # Chebs
        chebs_pixels, chebs_orders = get_chebyvals(pixel_centers_running, orders, max_pixel, max_order, deg_intra_order, deg_inter_order)
        
        # Loss
        loss = (coeffs) -> begin
            _model = build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, reshape(coeffs, (deg_inter_order+1, deg_intra_order+1)), orders)
            wres = weights_running .* (λ_centers_running .- _model)
            return wres
        end

        # Lsq
        result = scipyopt.least_squares(loss, p0, max_nfev=800 * length(coeffs_best), method="lm")
        coeffs_best .= result["x"]

        # Flag
        model_best = build_λsolution_chebyval2d_flat(chebs_pixels, chebs_orders, reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1)), orders)
        residuals = Maths.δλ2δv.(λ_centers .- model_best, λ_centers)
        σuse = findall(isfinite.(residuals) .&& (abs.(residuals) .> 0))
        bad = findall(abs.(residuals) .> min(3 * Maths.robust_stddev(residuals[σuse]), max_vel_cut))
        if length(bad) == 0
           break
        end
        weights_running[bad] .= 0
    end

    good_peaks = findall(weights_running .> 0)
    coeffs_best = reshape(coeffs_best, (deg_inter_order+1, deg_intra_order+1))

    # Return
    return coeffs_best, good_peaks
end

"""
get_chebyvals(pixels, orders, max_pixel::Real, max_order::Real, deg_intra_order::Int, deg_inter_order::Int)
"""
function get_chebyvals(pixels::AbstractVector{<:Real}, orders::AbstractVector{<:Real}, max_pixel::Real, max_order::Real, deg_intra_order::Int, deg_inter_order::Int)
    chebs_pixels = Vector{Float64}[]
    chebs_orders = Vector{Float64}[]
    @assert length(pixels) == length(orders)
    for i in eachindex(pixels)
        push!(chebs_pixels, Maths.chebyvals(pixels[i] / max_pixel, deg_intra_order))
        push!(chebs_orders, Maths.chebyvals(orders[i] / max_order, deg_inter_order))
    end
    return chebs_pixels, chebs_orders
end