function nextodd(x) 
    x = Int(round(x))
    x = isodd(x) ? x : x + 1
    @assert x > 0
    return x
end