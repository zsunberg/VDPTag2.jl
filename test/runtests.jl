using VDPTag2
using Base.Test
using POMDPToolbox
using MCTS
using POMDPs

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
