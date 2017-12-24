if !isdefined(:gym) include("../base.jl") end

import POMDPs: MDP, Solver

type CrossentropyPolicySolver <: Solver
    epochs::Int64
    n_samples::Int64
    percentile::Float64
    smoothing::Float64
    max_frame_iterations::Int64
    print_every_n::Int64
end

CrossentropyPolicySolver(epochs = 10, n_samples = 100, percentile = 0.5, smoothing = 0.1; max_frame_iterations = 100, print_every_n = 25) =
    CrossentropyPolicySolver(epochs, n_samples, percentile, smoothing, max_frame_iterations, print_every_n)

function solve(solver::CrossentropyPolicySolver, pomdp::MDP; verbose = true)

    policy = CrossentropyPolicy(pomdp)

    for i=1:solver.epochs

        batch_states  = Vector{Int64}[] # a list of lists of states in each session
        batch_actions = Vector{Int64}[] # a list of lists of actions in each session
        batch_rewards = Float64[] # a list of floats - total rewards at each session

        for j=1:solver.n_samples

            if verbose && (j == 1 || j % solver.print_every_n == 0)
                println("Epoch: $i, Episode: $j")
            end

            session_states, session_actions, session_reward = run_experiment(pomdp, policy, solver.max_frame_iterations; keep_history = true)
            push!(batch_states, session_states)
            push!(batch_actions, session_actions)
            append!(batch_rewards, session_reward)
        end

        # Identify threshold which filters out best performing games
        reward_threshold = quantile(batch_rewards, solver.percentile);
        reward_valid_indices = batch_rewards .> reward_threshold;

        # print("Max reward: $reward_threshold")

        # Create a filtered view to best peforming games
        filtered_states  = view(batch_states, reward_valid_indices);
        filtered_actions = view(batch_actions, reward_valid_indices);

        # Create a new policy matrix which has identical structure and smoothing value
        policy_values = zeros(policy.action_map) + solver.smoothing;
        for ep in 1:sum(reward_valid_indices)
            map(x -> policy_values[filtered_actions[ep][x], filtered_states[ep][x]] += 1 , 1:length(filtered_actions[ep]));
        end

        policy_values = policy_values ./ sum(policy_values, 1)
        policy = CrossentropyPolicy(pomdp, policy_values, Int64(round(mean(batch_rewards), 0)))
        # policy = CrossentropyPolicy(pomdp, policy_values, 0)

        print("Sum score: $(sum(batch_rewards)), Max score: $(maximum(batch_rewards)), Mean score: $(round(mean(batch_rewards), 2)), Median score: $(median(batch_rewards))")

    end

    return policy

end
