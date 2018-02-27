@recipe function f(p::VDPTagProblem)
    m = mdp(p)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    bs = p.barriers
    if bs isa CardinalBarriers
        for dir in cardinals() 
            ends = (bs.start*dir, (bs.start+bs.len)*dir)
            color := :black
            label --> ""
            @series [v[1] for v in ends], [v[2] for v in ends]
        end
    end
    nothing
end

@recipe function f(pomdp::VDPTagPOMDP, h::AbstractPOMDPHistory{TagState})
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    @series mdp(pomdp), h
    @series begin
        label := "belief"
        belief_hist(h)[end]
    end
end

@recipe function f(p::VDPTagProblem, h::SimHistory)
    m = mdp(p)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    @series begin
        label := "path"
        x = [s.agent[1] for s in state_hist(h)[1:end-1]]
        y = [s.agent[2] for s in state_hist(h)[1:end-1]]
        x, y
    end
    @series begin
        a = action_hist(h)[end]
        if a isa TagAction && a.look
            color := :blue
        else
            color := :red
        end
        s = state_hist(h)[end-1]
        label := "current agent position"
        pts = Plots.partialcircle(0, 2*pi, 100, m.tag_radius)
        x, y = Plots.unzip(pts)
        x+s.agent[1], y+s.agent[2]
    end
    @series begin
        seriestype := :scatter
        label := "current target"
        pos = state_hist(h)[end-1].target
        color --> :orange
        [pos[1]], [pos[2]]
    end
    @series begin m end
end


@recipe function f(pc::ParticleCollection{TagState})
    seriestype := :scatter
    x = [p.target[1] for p in particles(pc)]
    y = [p.target[2] for p in particles(pc)]
    color --> :black
    markersize --> 0.1
    x, y
end

"""
Create a gif of a history and return the filename.
"""
function gif(p::VDPTagProblem, h::SimHistory)
    
end
