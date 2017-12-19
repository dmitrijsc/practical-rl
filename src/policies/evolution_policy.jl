import POMDPs: MDP, Policy

#
# Define a policy that will act as an evolutionary/ genetic policy
#
type EvolutionPolicy <: Policy
    action_map::Vector{Int64}
    score::Int64
end

#
# Default random intializer based on a games input
#
function EvolutionPolicy(pomdp::MDP; rng::AbstractRNG = MersenneTwister(0))
    default_map = rand(actions(pomdp), n_states(pomdp)) # all zeros
    return EvolutionPolicy(default_map, 0)
end

#
# Default initializer with another policy as an input and a random
# mutate probability.
#
function EvolutionPolicy(pomdp::MDP, p1::EvolutionPolicy; p = 0.1, rng::AbstractRNG = MersenneTwister(0))
    p2 = EvolutionPolicy(pomdp; rng = rng)
    return EvolutionPolicy(pomdp, p1, p2; p = p, rng = rng)
end

#
# Default initializer that acts as a crossover between two different policies
#
function EvolutionPolicy(pomdp::MDP, p1::EvolutionPolicy, p2::EvolutionPolicy; p = 0.5, rng::AbstractRNG = MersenneTwister(0))

    l = n_states(pomdp)
    m = ifelse.(rand(rng, l) .< p, zeros(l), ones(l))

    new_action_map = m .* p1.action_map + (1 - m) .* p2.action_map

    return EvolutionPolicy(Vector{Int64}(new_action_map), 0)
end

#
# Select and action from a policy based on a state
#
action(policy::EvolutionPolicy, s::Int64) = policy.action_map[s]
