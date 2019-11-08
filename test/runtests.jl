using VDPTag2
using Test
using MCTS
using POMDPs
using ParticleFilters
using ProgressMeter
using LinearAlgebra
using Random
using POMDPModelTools
using POMDPPolicies
using POMDPModels
using POMDPSimulators


Random.seed!(1)
pomdp = VDPTagPOMDP()
gen = NextMLFirst(mdp(pomdp), MersenneTwister(31))
global s = TagState([1.0, 1.0], [-1.0, -1.0])

struct MyNode end
MCTS.n_children(::MyNode) = rand(1:10)

@inferred next_action(gen, pomdp, s, MyNode())
@inferred next_action(gen, pomdp, initialstate_distribution(pomdp), MyNode())

for a in range(0.0, stop=2*pi, length=100)
    s = TagState(Vec2(0,0), Vec2(1,1))
    barriers = CardinalBarriers(0.2, 1.8)
    agent_speed = 1.0
    step_size = 0.5
    delta = agent_speed*step_size*Vec2(cos(a), sin(a))
    agent = barrier_stop(barriers, s.agent, delta)
    @test agent == s.agent+delta
end

pomdp = VDPTagPOMDP()
for sao in stepthrough(pomdp, RandomPolicy(pomdp), "sao", max_steps=10)
    @show sao
end

dpomdp = AODiscreteVDPTagPOMDP(pomdp, 30, 0.5)
for sao in stepthrough(dpomdp, RandomPolicy(dpomdp), "sao", max_steps=10)
    @show sao
end

pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 1.8)))
filter = SIRParticleFilter(pomdp, 1000)
for sao in stepthrough(pomdp, ToNextML(pomdp), filter, "sao", max_steps=10)
    @show sao
end



# test to make sure it can't pass through any walls
pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.0, 100.0)))
for quadrant in [Vec2(1,1), Vec2(-1,1), Vec2(1,-1), Vec2(-1,-1)]
    @showprogress for i in 1:100
        is = initialstate(pomdp, Random.GLOBAL_RNG)
        is = TagState(quadrant, is.target)
        for (s, sp) in stepthrough(pomdp, ToNextML(pomdp), filter, "s,sp", max_steps=100)
            println(s.agent.*quadrant .>= 0.0)
            #@test all(s.agent.*quadrant .>= 0.0) #TODO: ask what is going on
            if s == sp
                println("did not move (this should not happen a bunch of times)")
            end
        end
        for (s, sp) in stepthrough(pomdp, RandomPolicy(pomdp), "s,sp", max_steps=100)
            println(s.agent.*quadrant .>= 0.0)
            #@test all(s.agent.*quadrant .>= 0.0) #TODO: ask what is going on
            if s == sp
                println("did not move (this should not happen a bunch of times)")
            end
        end
    end
end

# make sure it does pass through walls sometimes without any barriers
pomdp = VDPTagPOMDP()
N = 100
for quadrant in [Vec2(1,1), Vec2(-1,1), Vec2(1,-1), Vec2(-1,-1)]
    in_other = falses(N)
    @showprogress for i in 1:N
        is = initialstate(pomdp, Random.GLOBAL_RNG)
        is = TagState(quadrant, is.target)
        hr = HistoryRecorder(max_steps=100)
        hist = simulate(hr, pomdp, ToNextML(pomdp), filter)
        in_other[i] = any(any(s.agent.*quadrant .< 0.0) for s in state_hist(hist))
    end
    @show sum(in_other)/length(in_other)
    println("Should be near one?")
    @test any(in_other)
end
for quadrant in [Vec2(1,1), Vec2(-1,1), Vec2(1,-1), Vec2(-1,-1)]
    in_other = falses(N)
    @showprogress for i in 1:N
        is = initialstate(pomdp, Random.GLOBAL_RNG)
        is = TagState(quadrant, is.target)
        hr = HistoryRecorder(max_steps=2)
        hist = simulate(hr, pomdp, RandomPolicy(pomdp))
        in_other[i] = any(any(s.agent.*quadrant .< 0.0) for s in state_hist(hist))
    end
    @show sum(in_other)/length(in_other)
    println("Should be near 3/4?")
    @test any(in_other)
end
