module DW

using Agents
using DataFrames
using Statistics
using Plots


@agent daisy_patch GridAgent{2} begin
    colour::Int
    temperature::Float64
end

# Define a set of values that can be used to create a default model
size_dflt = 25
T_opt_dflt = 22.5
T_range_dflt = 10.0
P_death_dflt = 0.1
P_growth_dflt = 0.05
delta_dflt = 0.25
albedo_dflt = [0.75, 0.25, 0.5]
F_dflt(step) = (T_opt_dflt + T_range_dflt)*2.0

function create_properties(size_dflt, T_opt_dflt, T_range_dflt, P_death_dflt, P_growth_dflt, delta_dflt, albedo_dflt, F_dflt)
    β(T) = max(-(T-T_opt_dflt+T_range_dflt)*(T-T_opt_dflt-T_range_dflt)/T_range_dflt^2, 0.0)
    return Dict{Symbol,Any}(:T_opt => T_opt_dflt, 
    :T_range=>T_range_dflt, 
    :P_death=>P_death_dflt, 
    :P_growth=>P_growth_dflt, 
    :delta=>delta_dflt,
    :step=>0,
    :flux=>F_dflt,
    :growth_rate=>β,
    :albedo=>albedo_dflt
    )
end

properties_dflt = create_properties(size_dflt, T_opt_dflt, T_range_dflt, P_death_dflt, P_growth_dflt, delta_dflt, albedo_dflt, F_dflt)

function agent_step!(agent, model)
    if agent.colour == 3
        # With low probability an empty patch can grow daisies
        rand() <= model.P_growth && (agent.colour = rand([1,2]))
    else
        # An occupied patch can attempt to reproduce
        T = agent.temperature
        neighbour = rand(collect(nearby_agents(agent, model)))
        if neighbour.colour == 3
            rand() <= model.growth_rate(T) && (neighbour.colour = agent.colour)
        end
        
        # An occupied patch can die and revert to being empty.
        rand() <= model.P_death && (agent.colour = 3)
    end
end



function model_step!(model)
    t =  model.step
    δ = model.delta
    F = model.flux
    # Calculate the proportion of absorbed energy
    A = [1.0- x for x in model.albedo]
    
    # Calculate the temperature at each point in the space
    for pos in positions(model)
        agent = collect(agents_in_position(pos, model))[1]
        T1 = A[agent.colour]*F(t)
        neighbours = collect(nearby_agents(agent, model))
        T2=0.0
        for n in neighbours
            T2 += A[n.colour]*F(t)
        end
        T2 = T2/(length(neighbours))
        
        # Update the local temperature to be a weighted sum of T1 and T2
        agent.temperature = (1.0-δ)*T1 + δ*T2
    end
    
    model.step = t+1
end


function initialise(;dims = (size,size), properties=dflt_properties)
    # Create the underlying space
    space = GridSpace(dims, periodic = true)
    
    model = ABM(daisy_patch, space; scheduler = fastest, properties=properties)
    
    # Initial temperature is that of "bare" planet
    Tinit = properties[:flux](1)*properties[:albedo][3]
   
    # Add an agent to each node of the space. Initially all have type 3 (empty) and temperature 0
    for node in nodes(model)
        add_agent!(node, model, 3, Tinit)
    end
    
    return model
end

function run(nsteps; size=size_dflt, T_opt=T_opt_dflt, T_range=T_range_dflt, P_death=P_death_dflt, 
        P_growth=P_growth_dflt, delta=delta_dflt, albedo=albedo_dflt, F=F_dflt)
    
    properties = create_properties(size, T_opt, T_range, P_death, P_growth, delta, albedo, F)
    
    model = initialise(dims = (size,size), properties=properties)
    
    adata = [:pos, :colour, :temperature]
    when = 1:nsteps  # At which steps to collect data
    data, _  = run!(model, agent_step!, model_step!, nsteps; adata);
    
    gdf = groupby(data, :step)
    df_T = combine(gdf, :temperature => mean => :T_average)
    
    gdf = groupby(data, [:step, :colour]);
    df = combine(gdf, nrow => :count);
    colour_df = groupby(df, :colour);
    g1 = get(colour_df, (colour=1,), nothing);
    g2 = get(colour_df, (colour=2,), nothing);
    g3 = get(colour_df, (colour=3,), nothing);
    
    p = plot(df_T.step, df_T.T_average /T_opt, label = "T/T_opt", color="red")
    plot!(g1.step, g1.count/size^2, label = "Proportion of white daisies", color="green")
    plot!(g2.step, g2.count/size^2, label = "Proportion of black daisies", color="black")
    plot!(g3.step, g3.count/size^2, label = "Proportion of empty patches", color="blue")
    
    return data, p
end

end
    
    
