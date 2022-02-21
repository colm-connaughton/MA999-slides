module Schelling

using Agents
using Random # for reproducibility

@agent SchellingAgent GridAgent{2} begin
    mood::Bool
    group::Int
    neighbours::Int
    interfaces::Float64
end


function initialize(; numagents = 320, griddims = (20, 20), min_to_be_happy = 3, seed = 125)
    space = GridSpace(griddims, periodic = false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.MersenneTwister(seed)
    model = ABM(
        SchellingAgent, space;
        properties, rng, scheduler = Schedulers.randomly
    )

    # populate the model with agents, adding equal amount of the two types of agents
    # at random positions in the model
    for n in 1:numagents
        agent = SchellingAgent(n, (1, 1), false, n < numagents / 2 ? 1 : 2, 0, 0.0)
        add_agent_single!(agent, model)
    end
    return model
end

function agent_step!(agent, model)
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
    # For each neighbor, get group and compare to current agent's group
    # and increment count_neighbors_same_group as appropriately.
    # Here `nearby_agents` (with default arguments) will provide an iterator
    # over the nearby agents one grid point away, which are at most 8.
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    # After counting the neighbors, decide whether or not to move the agent.
    # If count_neighbors_same_group is at least the min_to_be_happy, set the
    # mood to true. Otherwise, move the agent to a random position.
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        move_agent_single!(agent, model)
    end
    return
end

function model_step!(model)
    # this function updates the total number of neighbours and the number of unlike 
    # neighbours of each agent after the position updates are complete
    for agent in allagents(model)
        count_neighbors_other_group = 0
        count_total_neighbours = 0 
        for neighbor in nearby_agents(agent, model)
            if agent.group != neighbor.group
                count_neighbors_other_group += 1
            end
            count_total_neighbours += 1
        end
        agent.neighbours = count_total_neighbours
        if count_total_neighbours >0
            agent.interfaces = count_neighbors_other_group/count_total_neighbours
        else
            agent.interfaces=0.0
        end

    end
end

end