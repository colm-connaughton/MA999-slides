module CG

using Agents
using DataFrames
using Plots
using Statistics

include("all_simple_paths.jl")

# Create a basic setup that can be used to create a default model

# Make a default graph
dflt_G = DiGraph(4) # graph with 4 vertices
add_edge!(dflt_G, 1, 2)
add_edge!(dflt_G, 1, 3)
add_edge!(dflt_G, 2, 4)
add_edge!(dflt_G, 3, 4)

# Set default link parameters 
dflt_link_parameters = [(0.1, 0.0), (0.0, 15.0), (0.0, 15.0), (0.1, 0.0)];

# Set default temperature
dflt_T = 50.0



# Function to determine if a given link is on a given path
function link_path_adjacency(link, path)
    for i in 1:(length(path)-1)
        (link.src == path[i] && link.dst == path[i+1]) && return 1
    end
    return 0
end

function create_properties(G::SimpleDiGraph{Int64}, link_parameters::Array{Tuple{Float64,Float64},1}, T::Float64)
    # Use these default link parameters to define demand-delay functions for each link
    n = length(link_parameters)
    a1 = collect(1:n)
    a2 = [x ->  link_parameters[i][1] * x + link_parameters[i][2] for i in 1:n]
    demand_delay_functions = Dict( a1 .=> a2)

    # Get all paths linking node 1 to node 4
    paths = all_simple_paths(G, 1, 4)

    # Use this function to calculate default link path adjacency matrix
    E = collect(edges(G))
    A = zeros(Int64, (length(E), length(paths)))
    for (i, edge) in enumerate(E), (j, path) in enumerate(paths)
        A[i,j] = link_path_adjacency(edge, path)
    end

    # Initially route times are set to zero
    route_times = zeros(Float64,length(paths))
    
    # Put all of these together to define a properties array
    return Dict{Symbol,Any}(:T => T, :link_path_matrix=>A, :demand_delay=>demand_delay_functions, :route_times=>route_times, :routes => paths)
end

dflt_properties = create_properties(dflt_G, dflt_link_parameters, dflt_T)

mutable struct driverAgent <: AbstractAgent
    id::Int
    route_choice::Int
    travel_time::Float64
end

function initialise(;numagents = 100, properties = dflt_properties)
    model = ABM(driverAgent, scheduler=fastest, properties=properties)
    for i in 1:numagents
        add_agent!(model, 1, 0.0)
    end
    return model
end

function agent_step!(agent, model)
    T = model.properties[:T]
    previous_route_times = model.properties[:route_times]
    nroutes = length(previous_route_times)
    # Check how long agent's current route choice took the previous day
    t = previous_route_times[agent.route_choice]
    
    # Make a list of alternative routes
    alternative_routes = collect(1:nroutes)
    deleteat!(alternative_routes, alternative_routes .== agent.route_choice)
    
    # Check how long a random alternative would have taken
    alternative_route = rand(alternative_routes)
    t_alternative = previous_route_times[alternative_route]
    
    # Calculate the difference between the times for chosen route and alternative
    Δt = t_alternative - t
    # Switch to the alternative route if Δt is negative with a probability that tends to
    # zero as Δt tends to zero
    if Δt <=0
        P = 1.0 - exp(Δt/T)
        rand() < P && (agent.route_choice = alternative_route)
    end
end

function model_step!(model)
    A = model.properties[:link_path_matrix]
    nlinks=size(A)[1]
    npaths=size(A)[2]
    
    # Count how many agents chose each route
    path_counts = zeros(Int64,npaths)
    for agent in allagents(model)
        r = agent.route_choice
        path_counts[r]+=1
    end
    
    # Derive the link counts from the path counts
    link_counts = A * path_counts
    
    # Derive the link travel times from the link counts
    link_travel_times = [model.properties[:demand_delay][i](link_counts[i]) for i in 1:nlinks]
    
    # Calculate updated path travel times for each route based on the link travel times
    model.properties[:route_times] = link_travel_times'*A
    
    # Update the agents travel times
    for agent in allagents(model)
        r = agent.route_choice
        agent.travel_time = model.properties[:route_times][r]
    end
end

function run(;G=dflt_G, link_parameters = dflt_link_parameters, T = dflt_T, numagents=100, numsteps=100)
    # Build the properties array
    properties =  create_properties(G, link_parameters, T)
    # Create the model
    model = initialise(numagents = numagents, properties = properties)
    # Specify data collection
    adata = [:route_choice, :travel_time]
    # Run the model
    data, _  = run!(model, agent_step!, model_step!, numsteps; adata);
    # Post-process the model output
    # Plot the average travel time vs model step
    gdf = groupby(data, :step)
    df_average_travel = combine(gdf, :travel_time => mean => :T)
    delete!(df_average_travel, 1)
    p1 = plot(df_average_travel.step, df_average_travel.T, ylims=[0.0, 50.0], label="Average travel time")
    xlabel!("Steps")
    ylabel!("Time")
    # Plot the number of agents choosing each path vs model step
    gdf = groupby(data, [:step, :route_choice])
    df = combine(gdf, nrow => :count)
    routes_df = groupby(df, :route_choice)
    p2 = plot()
    paths = collect(model.properties[:routes])
    for (key, df) in pairs(routes_df)
        idx = key.route_choice
        label = "Route "*string(idx)*" : "*string(paths[idx])
        plot!(df.step, df.count, label=label )
    end
    xlabel!("Steps")
    ylabel!("Users")
    
    return data, p1, p2
end

end