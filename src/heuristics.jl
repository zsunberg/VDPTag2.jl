immutable ToNextML <: Policy
    p::VDPTagMDP
    rng::MersenneTwister
end

ToNextML(p::VDPTagProblem; rng=Base.GLOBAL_RNG) = ToNextML(mdp(p), rng)

function action(p::ToNextML, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next-s.agent
    return atan2(diff[2], diff[1])
end

action(p::ToNextML, b::ParticleCollection{TagState}) = TagAction(false, action(p, rand(p.rng, b)))

immutable ToNextMLSolver <: Solver
    rng::MersenneTwister
end

solve(s::ToNextMLSolver, p::VDPTagProblem) = ToNextML(mdp(p), s.rng)
function solve(s::ToNextMLSolver, dp::DiscreteVDPTagProblem)
    cp = cproblem(dp)
    return translate_policy(ToNextML(mdp(cp), s.rng), cp, dp, dp)
end


immutable ManageUncertainty <: Policy
    p::VDPTagPOMDP
    max_norm_std::Float64
end

function action(p::ManageUncertainty, b::ParticleCollection{TagState})
    agent = first(particles(b)).agent
    target_particles = Array{Float64}(2, n_particles(b))
    for (i, s) in enumerate(particles(b))
        target_particles[:,i] = s.target
    end
    normal_dist = fit(MvNormal, target_particles)
    angle = action(ToNextML(mdp(p.p)), TagState(agent, mean(normal_dist)))
    return TagAction(sqrt(det(cov(normal_dist))) > p.max_norm_std, angle)
end

type NextMLFirst{RNG<:AbstractRNG}
    p::VDPTagMDP
    rng::RNG
end

function next_action(gen::NextMLFirst, mdp::Union{POMDP, MDP}, s::TagState, snode)
    if n_children(snode) < 1
        return action(ToNextML(gen.p, gen.rng), s)
    else
        return 2*pi*rand(gen.rng)
    end
end

function next_action(gen::NextMLFirst, pomdp::Union{POMDP, MDP}, b, onode)
    s = rand(gen.rng, b)
    ca = TagAction(false, next_action(gen, pomdp, s, onode))
    return convert_a(action_type(pomdp), ca, pomdp)
end

immutable TranslatedPolicy{P<:Policy, T, ST, AT} <: Policy
    policy::P
    translator::T
    S::Type{ST}
    A::Type{AT}
end

function translate_policy(p::Policy, from::Union{POMDP,MDP}, to::Union{POMDP,MDP}, translator)
    return TranslatedPolicy(p, translator, state_type(from), action_type(to))
end

function action(p::TranslatedPolicy, s)
    cs = convert_s(p.S, s, p.translator)
    ca = action(p.policy, cs)
    return convert_a(p.A, ca, p.translator)
end

# this is not the most efficient way to do this
function action(p::TranslatedPolicy, pc::AbstractParticleBelief)
    @assert !isa(pc, WeightedParticleBelief)
    cpc = ParticleCollection([convert_s(p.S, s, p.translator) for s in particles(pc)])
    ca = action(p.policy, cpc)
    return convert_a(p.A, ca, p.translator)
end
