@recipe function f(mdp::VDPTagMDP, s::TagState)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    @series begin
        color := :black
        seriestype := :scatter
        label := "target"
        markersize := 0.1
        [s.target[1]], [s.target[2]]
    end
    @series begin
        color --> :blue
        label --> "agent"
        pts = Plots.partialcircle(0, 2*pi, 100, mdp.tag_radius)
        x, y = Plots.unzip(pts)
        x+s.agent[1], y+s.agent[2]
    end
end


@recipe function f(pomdp::VDPTagPOMDP, h::AbstractPOMDPHistory{TagState})
    mdp = pomdp.mdp
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
        if action_hist(h)[end].look
            color := :blue
        else
            color := :red
        end
        s = state_hist(h)[end-1]
        label := "current agent position"
        pts = Plots.partialcircle(0, 2*pi, 100, mdp.tag_radius)
        x, y = Plots.unzip(pts)
        x+s.agent[1], y+s.agent[2]
    end
    @series begin
        label := "belief"
        belief_hist(h)[end]
    end
    @series begin
        seriestype := :scatter
        label := "current target"
        pos = state_hist(h)[end-1].target
        color --> :orange
        [pos[1]], [pos[2]]
    end
end

@recipe function f(pc::ParticleCollection{TagState})
    seriestype := :scatter
    x = [p.target[1] for p in particles(pc)]
    y = [p.target[2] for p in particles(pc)]
    color --> :black
    markersize --> 0.1
    x, y
end
