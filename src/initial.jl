immutable VDPInitDist end
sampletype(::Type{VDPInitDist}) = TagState
function rand(rng::AbstractRNG, d::VDPInitDist)
    return TagState([0.0, 0.0], 8.0*rand(rng, 2)-4.0)
end

initial_state_distribution(::VDPTagProblem) = VDPInitDist()
