if !isdefined(:gym) include("../base.jl") end

import POMDPs: MDP, Solver

type CrossentropyPolicySolver <: Solver
    epochs::Int64
    n_samples::Int64
    percentile::Float64
    smoothing::Float64
    max_frame_iterations::Int64
    min_score::Float64
    min_score_step::Float64
    print_every_n::Int64
end

CrossentropyPolicySolver(epochs = 10, n_samples = 100, percentile = 0.5, smoothing = 0.1; max_frame_iterations = 100, print_every_n = 25, min_score = 0.0, min_score_step = 0.0) =
    CrossentropyPolicySolver(epochs, n_samples, percentile, smoothing, max_frame_iterations, min_score, min_score_step, print_every_n)

function solve(solver::CrossentropyPolicySolver, pomdp::MDP; verbose = true)

    policy = CrossentropyPolicy(pomdp)

    last_value = -1

    for i=1:solver.epochs

        batch_states  = Vector{Int64}[] # a list of lists of states in each session
        batch_actions = Vector{Int64}[] # a list of lists of actions in each session
        batch_rewards = Float64[] # a list of floats - total rewards at each session

        for j=1:solver.n_samples

            if verbose && (j == 1 || j % solver.print_every_n == 0)
                println("Epoch: $i, Episode: $j")
            end

            session_states, session_actions, session_reward, session_states_new, session_rewards = run_experiment(pomdp, policy, solver.max_frame_iterations; keep_history = true)
            push!(batch_states, session_states)
            push!(batch_actions, session_actions)
            append!(batch_rewards, session_reward)
        end

        # Identify threshold which filters out best performing games
        reward_threshold = quantile(batch_rewards, solver.percentile);
        reward_threshold += if reward_threshold == solver.min_score solver.min_score_step else zero(reward_threshold) end
        reward_valid_indices = batch_rewards .>= reward_threshold;
        reward_valid_indices_count = sum(reward_valid_indices)

        # Create a filtered view to best peforming games
        filtered_states  = view(batch_states, reward_valid_indices);
        filtered_actions = view(batch_actions, reward_valid_indices);

        # Create a new policy matrix which has identical structure and smoothing value
        if reward_valid_indices_count > 0

            policy_values = zeros(policy.action_map) + solver.smoothing;
            for ep in 1:reward_valid_indices_count
                map(x -> policy_values[filtered_actions[ep][x], filtered_states[ep][x]] += 1 , 1:length(filtered_actions[ep]));
            end

            policy_values = policy_values ./ sum(policy_values, 1)
            policy.action_map = policy_values

        end

        println("Threshold: $reward_threshold, Size: $(sum(reward_valid_indices))")
        println("Sum score: $(sum(batch_rewards)), Max score: $(maximum(batch_rewards)), Mean score: $(round(mean(batch_rewards), 2)), Median score: $(median(batch_rewards))")

    end

    return policy

end
