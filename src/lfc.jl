export get_lfc_modes

function get_lfc_mode_pixel_spacing(xi, xf, λi, λf, ν0, Δν)
    integers, _ = gen_lfc_modes(λi, λf, ν0, Δν)
    s = (xf - xi) / length(integers)
    return s
end

function get_lfc_modes(λ, spec, ν0, Δν; σ_bounds=[0.2, 4.0], xrange=nothing, background_poly_deg=0, min_mode_spacing=nothing)

    # Resolve xrange
    xrange = resolve_xrange(spec, xrange)

    # Get mode spacing in pixels
    if isnothing(min_mode_spacing)
        min_mode_spacing = 0.5 * get_lfc_mode_pixel_spacing(xrange[1], xrange[2], λ[xrange[1]], λ[xrange[2]], ν0, Δν)
    end

    # Get locations in pixel space
    modes_pixels, amplitudes, σs, background_polys, rms = get_modes(spec, min_mode_spacing; σ_bounds, xrange, background_poly_deg)

    # Pair with known mode numbers
    mode_integers, mode_λs = pair_lfc_modes(modes_pixels, λ, ν0, Δν)

    # Return
    return modes_pixels, mode_λs, mode_integers, amplitudes, σs, background_polys, rms

end


function pair_lfc_modes(modes_pixels, λ, ν0, Δν)
    xmin, xmax = Int(floor(modes_pixels[1])), Int(ceil(modes_pixels[end]))
    λi, λf = λ[xmin], λ[xmax]
    integers, modes_λ = gen_lfc_modes(λi - 5, λf + 5, ν0, Δν)
    integers_true = zeros(Int, length(modes_pixels))
    modes_λ_true = fill(NaN, length(modes_pixels))
    for i in eachindex(modes_pixels)
        λ_approx = λ[Int(round(modes_pixels[i]))]
        k = argmin(abs.(modes_λ .- λ_approx))
        integers_true[i] = integers[k]
        modes_λ_true[i] = modes_λ[k]
    end
    @assert length(unique(integers_true)) == length(integers_true)
    return integers_true, modes_λ_true
end


function gen_lfc_modes(λi::Real, λf::Real, ν0::Real, Δν::Real)
    νf = (SPEED_OF_LIGHT_MPS / λi) * 1E9
    νi = (SPEED_OF_LIGHT_MPS / λf) * 1E9
    λ0 = (SPEED_OF_LIGHT_MPS / ν0) * 1E9
    n_left = max(Int(round(2 * (ν0 - νi) / Δν)), 0)
    n_right = max(Int(round(2 * (νf - ν0) / Δν)), 0)
    modes_ν = [ν0 - n_left * Δν:Δν:ν0 + n_right * Δν;]
    integers = [-n_left:n_right;]

    # Convert to wavelength
    modes_λ = SPEED_OF_LIGHT_MPS ./ modes_ν .* 1E9
    reverse!(modes_λ)
    reverse!(integers)

    good = findall(modes_λ .> λi .&& modes_λ .< λf)
    integers, modes_λ = integers[good], modes_λ[good]

    # Return
    return integers, modes_λ
end

function gen_lfc_modes(ν0::Real, Δν::Real, ints::AbstractVector{Int})
    modes_ν = ints .* Δν .+ ν0
    modes_λ = SPEED_OF_LIGHT_MPS ./ modes_ν .* 1E9
    return modes_λ
end