const IVec8 = SVector{8, Int}

@with_kw struct AODiscreteVDPTagPOMDP <: POMDP{TagState, Int, IVec8}
    cpomdp::VDPTagPOMDP = VDPTagPOMDP()
    n_angles::Int       = 10
    binsize::Float64    = 0.5
end

@with_kw struct ADiscreteVDPTagPOMDP <: POMDP{TagState, Int, Vec8}
    cpomdp::VDPTagPOMDP = VDPTagPOMDP()
    n_angles::Int       = 10
end


const DiscreteVDPTagProblem = Union{AODiscreteVDPTagPOMDP, ADiscreteVDPTagPOMDP}

cproblem(p::AODiscreteVDPTagPOMDP) = p.cpomdp
cproblem(p::ADiscreteVDPTagPOMDP) = p.cpomdp

convert_s(::Type{T}, x::T, p) where T = x
convert_a(::Type{T}, x::T, p) where T = x
convert_o(::Type{T}, x::T, p) where T = x

# state
function convert_s(::Type{Int}, s::TagState, p::DiscreteVDPTagProblem)
    n = p.n_bins
    factor = n/(2*p.grid_lim)
    ai = clamp(ceil(Int, (s.agent[1]+p.grid_lim)*factor), 1, n)
    aj = clamp(ceil(Int, (s.agent[2]+p.grid_lim)*factor), 1, n)
    ti = clamp(ceil(Int, (s.target[1]+p.grid_lim)*factor), 1, n)
    tj = clamp(ceil(Int, (s.target[2]+p.grid_lim)*factor), 1, n)
    return sub2ind((n,n,n,n), ai, aj, ti, tj)
end
function convert_s(::Type{TagState}, s::Int, p::DiscreteVDPTagProblem)
    n = p.n_bins
    factor = 2*p.grid_lim/n
    ai, aj, ti, tj = ind2sub((n,n,n,n), s)
    return TagState((Vec2(ai, aj)-0.5)*factor-p.grid_lim, (Vec2(ti, tj)-0.5)*factor-p.grid_lim)
end

# action
function convert_a(::Type{Int}, a::Float64, p::DiscreteVDPTagProblem)
    i = ceil(Int, a*p.n_angles/(2*pi))
    while i > p.n_angles
        i -= p.n_angles
    end
    while i < 1
        i += p.n_angles
    end
    return i
end
convert_a(::Type{Float64}, a::Int, p::DiscreteVDPTagProblem) = (a-0.5)*2*pi/p.n_angles

function convert_a(T::Type{Int}, a::TagAction, p::DiscreteVDPTagProblem)
    i = convert_a(T, a.angle, p)
    if a.look
        return i + p.n_angles
    else
        return i
    end
end
function convert_a(::Type{TagAction}, a::Int, p::DiscreteVDPTagProblem)
    return TagAction(a > p.n_angles, convert_a(Float64, a % p.n_angles, p))
end

# observation
function convert_o(::Type{IVec8}, o::Vec8, p::AODiscreteVDPTagPOMDP)
    return floor.(Int, (o./p.binsize)::Vec8)::IVec8
end
# convert_o(::Type{Vec8}, o::Int, p::DiscreteVDPTagProblem) = (o-0.5)*2*pi/p.n_obs_angles

n_states(p::AODiscreteVDPTagPOMDP) = Inf
n_actions(p::DiscreteVDPTagProblem) = 2*p.n_angles
discount(p::DiscreteVDPTagProblem) = discount(cproblem(p))
isterminal(p::DiscreteVDPTagProblem, s) = isterminal(cproblem(p), convert_s(TagState, s, p))

actions(p::DiscreteVDPTagProblem) = 1:n_actions(p)

function generate_s(p::DiscreteVDPTagProblem, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(action_type(cproblem(p)), a, p)
    return generate_s(cproblem(p), s, ca, rng)
end

function generate_sr(p::DiscreteVDPTagProblem, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(action_type(cproblem(p)), a, p)
    sp = generate_s(cproblem(p), s, ca, rng)
    r = reward(cproblem(p), s, ca, sp)
    return (sp, r)
end

function generate_sor(p::ADiscreteVDPTagPOMDP, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(action_type(cproblem(p)), a, p)
    return generate_sor(cproblem(p), s, ca, rng)
end

function generate_o(p::ADiscreteVDPTagPOMDP, s::TagState, a::Int, sp::TagState, rng::AbstractRNG)
    ca = convert_a(action_type(cproblem(p)), a, p)
    return generate_o(cproblem(p), s, ca, sp, rng)
end

function generate_sor(p::AODiscreteVDPTagPOMDP, s::TagState, a::Int, rng::AbstractRNG)
    ca = convert_a(action_type(cproblem(p)), a, p)
    csor = generate_sor(cproblem(p), s, ca, rng)
    return (csor[1], convert_o(IVec8, csor[2], p), csor[3])
end

function generate_o(p::AODiscreteVDPTagPOMDP, s::TagState, a::Int, sp::TagState, rng::AbstractRNG)
    ca = convert_a(action_type(cproblem(p)), a, p)
    co = generate_o(cproblem(p), s, ca, sp, rng)
    return convert_o(IVec8, co, p)
end

initialstate_distribution(p::AODiscreteVDPTagPOMDP) = VDPInitDist()
initialstate_distribution(p::ADiscreteVDPTagPOMDP) = VDPInitDist()

#=
gauss_cdf(mean, std, x) = 0.5*(1.0+erf((x-mean)/(std*sqrt(2))))
function obs_weight(p::AODiscreteVDPTagPOMDP, a::Int, sp::TagState, o::Int)
    cp = cproblem(p)
    @assert cp.bearing_std <= 2*pi/6.0 "obs_weight assumes σ <= $(2*pi/6.0)"
    ca = convert_a(action_type(cp), a, p)
    co = convert_o(obs_type(cp), o, p) # float between 0 and 2pi
    upper = co + 0.5*2*pi/p.n_angles
    lower = co - 0.5*2*pi/p.n_angles
    if ca.look
        diff = sp.target - sp.agent
        bearing = atan2(diff[2], diff[1])
        # three cases: o is in bin, below, or above
        if bearing <= upper && bearing > lower
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            prob = cdf_up - cdf_low
        elseif bearing <= lower
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            below_cdf_up = gauss_cdf(bearing, cp.bearing_std, upper-2*pi)
            below_cdf_low = gauss_cdf(bearing, cp.bearing_std, lower-2*pi)
            prob = cdf_up - cdf_low + below_cdf_up - below_cdf_low
        else # bearing > upper
            cdf_up = gauss_cdf(bearing, cp.bearing_std, upper)
            cdf_low = gauss_cdf(bearing, cp.bearing_std, lower)
            above_cdf_up = gauss_cdf(bearing, cp.bearing_std, upper+2*pi)
            above_cdf_low = gauss_cdf(bearing, cp.bearing_std, lower+2*pi)
            prob = cdf_up - cdf_low + above_cdf_up - above_cdf_low
        end
        return prob
    else
        return 1.0
    end
end

function obs_weight(p::ADiscreteVDPTagPOMDP, a::Int, sp::TagState, o::Float64)
    ca = convert_a(TagAction, a, p)
    return obs_weight(cproblem(p), ca, sp, o)
end
=#
