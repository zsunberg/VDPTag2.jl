using VDPTag2
using Base.Test
using POMDPToolbox
using MCTS
using POMDPs
using ParticleFilters
using ProgressMeter

srand(1)
pomdp = VDPTagPOMDP()
gen = NextMLFirst(mdp(pomdp), MersenneTwister(31))
s = TagState([1.0, 1.0], [-1.0, -1.0])

struct MyNode end
MCTS.n_children(::MyNode) = rand(1:10)

@inferred next_action(gen, pomdp, s, MyNode())
@inferred next_action(gen, pomdp, initial_state_distribution(pomdp), MyNode())

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
pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(1.0, 100.0)))
for quadrant in [Vec2(1,1), Vec2(-1,1), Vec2(1,-1), Vec2(-1,-1)]
    @showprogress for i in 1:100
        is = initial_state(pomdp, Base.GLOBAL_RNG)
        is = TagState(quadrant, is.target)
        for (s, sp) in stepthrough(pomdp, ToNextML(pomdp), filter, "s,sp", max_steps=100, initial_state=is)
            @test all(s.agent.*quadrant .>= 0.0)
            if s == sp
                println("did not move (this should not happen a bunch of times)")
            end
        end
        for (s, sp) in stepthrough(pomdp, RandomPolicy(pomdp), "s,sp", max_steps=100, initial_state=is)
            @test all(s.agent.*quadrant .>= 0.0)
            if s == sp
                println("did not move (this should not happen a bunch of times)")
            end
        end
    end
end
