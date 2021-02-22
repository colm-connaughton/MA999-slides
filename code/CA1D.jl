module CA1D

using Agents
using Plots
using Colors

@agent CA1DAgent GridAgent{1} begin
    state::Int
end

rule110 = Dict("111"=>0, "110"=>1, "101"=>1, "100"=>0, "011"=>1,
            "010"=>1, "001"=>1, "000"=>0)

rule22 = Dict("111"=>0, "110"=>0, "101"=>0, "100"=>1, "011"=>0,
            "010"=>1, "001"=>1, "000"=>0)


function initialise(;rules = rule110, gridsize = 100, initial_condition="singleton")
    space = GridSpace((gridsize,), metric=:euclidean, periodic = true)
    properties = Dict(:rules => rules)
    model = ABM(CA1DAgent, space; properties)
    
    for idx in 1:gridsize
        add_agent_pos!(CA1DAgent(idx, (idx,), 0), model)
    end
    
    if initial_condition == "singleton"
        model.agents[gridsize].state = 1
    else
        for idx in 1:gridsize
            model.agents[idx].state = rand([0,1])
        end
    end
    return model
end

function agent_step!(agent, model)
    neighbourhood = nearby_ids(agent.pos, model)
    str = ""
    for idx in neighbourhood
        str=str*string(model.agents[idx].state)
    end
    agent.state = model.rules[str]
end

function model_step!(model)
    new_states =  fill(0, nagents(model))
    for agent in allagents(model)
        neighbourhood = nearby_ids(agent.pos, model)
        str = ""
        for idx in neighbourhood
            str=str*string(model.agents[idx].state)
            #println(idx, " ", model.agents[idx].state)
        end
        new_states[agent.id] = model.rules[str]
    end
    
    # Now that we have worked out all the new states, overwrite the old states
    for k in keys(model.agents)
        model.agents[k].state = new_states[k]
    end
end

function run(;rules = rule110, gridsize = 100, initial_condition="singleton", nsteps=20, update="synchronous")
    # Initialise a new ABM
    model = initialise(rules=rules, gridsize=gridsize, initial_condition=initial_condition)
    # Specify what data to record from the simulation
    adata = [:pos, :state]
    # Run the model
    if update == "synchronous"
        data, _ = run!(model, dummystep, model_step!, nsteps; adata)
    else
        data, _ = run!(model, agent_step!, nsteps; adata)
    end
    # Process the data into a simple array for plotting
    A = zeros(Int64, (nsteps+1, nagents(model)))
    for i in 0:nsteps
        snapshot = data[data[!,:step] .== i, :]
        for j in Iterators.flatten(snapshot.pos)
            A[i+1,j] = snapshot.state[j]
        end
    end
    
    # Reverse the image to make it plot nicer
    B = map( x -> x == 0 ? 1 : 0, A)
    p = plot(Gray.(B), aspect_ratio = :equal, ylim=(0,nsteps+1), axis=nothing, xaxis=false, yaxis=false)
    
    return p, A
end

end
