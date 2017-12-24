using StatsBase
import POMDPs: MDP, Policy

#
# Define a policy that will act as an crossentropy policy
#
type CrossentropyPolicy <: Policy
    action_map::Array{Float64,2}
    action_space::Vector{Int64}
    score::Int64
end

#
# Default uniform intializer based on a games input
#
function CrossentropyPolicy(pomdp::MDP)
    action_count = n_actions(pomdp)
    default_map = ones(action_count, n_states(pomdp)) * 1.0 / action_count
    return CrossentropyPolicy(default_map, 1:action_count, 0)
end

#
# Initialize policy with a predefined values
#
function CrossentropyPolicy(pomdp::MDP, values::Array{Float64,2}, score::Int64 = 0)
    action_count = n_actions(pomdp)
    return CrossentropyPolicy(values, 1:action_count, score)
end

#
# Select random action from a policy based on a state and action weight
#
action(policy::CrossentropyPolicy, s::Int64) = sample(policy.action_space, Weights(policy.action_map[:, s]))
