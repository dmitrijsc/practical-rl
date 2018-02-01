using StatsBase
import POMDPs: MDP, Policy

#
# Define a policy that will act as an Sarsa policy
#
type SarsaPolicy <: Policy
    value_map::Array{Float64,2}
    action_space::Vector{Int64}
    epsilon::Float64
    learning_rate::Float64
    discount::Float64
end

#
# Default uniform intializer based on a games input
#
function SarsaPolicy(pomdp::MDP, epsilon::Float64, learning_rate::Float64, discount::Float64)
    action_count = n_actions(pomdp)
    default_value_map = zeros(Float64, action_count, n_states(pomdp))
    return SarsaPolicy(default_value_map, 1:action_count, epsilon, learning_rate, discount)
end

#
# Select random action from a policy based on a state and action weight
#
function action(policy::SarsaPolicy, s::Int64)

    value, index = findmax(policy.value_map[:, s])
    is_uniform = iszero(policy.value_map[:, s])

    return if policy.epsilon < rand() && ~is_uniform policy.action_space[index] else rand(policy.action_space) end
end
