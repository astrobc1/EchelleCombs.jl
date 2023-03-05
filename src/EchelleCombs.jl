module EchelleCombs

const SPEED_OF_LIGHT_MPS = 299792458.0

function resolve_xrange(x, xrange=nothing)
    good = findall(isfinite.(x))
    if length(good) == 0
        return Int[]
    elseif length(good) == 1
        return [good[1], good[1]]
    else
        if !isnothing(xrange)
            xrange = Int.(xrange)
            xi, xf = max(xrange[1], good[1]), min(xrange[2], good[end])
        else
            xi, xf = good[1], good[end]
        end
        return [xi, xf]
    end
end


include("mode_fitting.jl")


include("peak_finding.jl")


include("lfc.jl")


include("drifts.jl")


include("utils.jl")


end
