using VDPTag2
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using ParticleFilters

frames = Frames(MIME("image/png"), fps=2)

pomdp = VDPTagPOMDP()
policy = ManageUncertainty(pomdp, 0.01)
# policy = ToNextML(mdp(pomdp))

rng = MersenneTwister(4)

hr = HistoryRecorder(max_steps=100, rng=rng)
filter = SIRParticleFilter(pomdp, 1000, rng=rng)
hist = simulate(hr, pomdp, policy, filter)

gr()
@showprogress "Creating gif..." for i in 1:n_steps(hist)
    push!(frames, plot(pomdp, view(hist, 1:i)))
end

filename = string(tempname(), "_vdprun.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
