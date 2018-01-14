import POMDPs: MDP, Solver

type ValueTablePolicySolver <: Solver
    epochs::Int64
    epsilon::Float64
    learning_rate::Float64
    discount::Float64
    max_frame_iterations::Int64
    print_every_n::Int64
end

ValueTablePolicySolver(epochs; epsilon = 0.2, learning_rate = 0.8, discount = 0.8, max_frame_iterations = 100, print_every_n = 25) =
    ValueTablePolicySolver(epochs, epsilon, learning_rate, discount, max_frame_iterations, print_every_n)

function solve(solver::ValueTablePolicySolver, pomdp::MDP; verbose = true)

    policy = ValueTablePolicy(pomdp, solver.epsilon, solver.learning_rate, solver.discount)
    last_rewards = zeros(Float64, solver.epochs)

    for i=1:solver.epochs

        if verbose && (i == 1 || i % solver.print_every_n == 0)
            println("Epoch: $i")
        end

        states_a, state_actions, state_reward, states_b, state_rewards = run_experiment(pomdp, policy, solver.max_frame_iterations; keep_history = true)

        for state_index in 1:length(states_a)

            current_action = state_actions[state_index]
            current_state = states_a[state_index]

            #println("State index: $state_index, Current state: $current_action, Current actions: $current_action")
            #println("Size: $(length(state_rewards)), $(states_b[state_index])")

            next_value = if isterminal(pomdp, states_b[state_index]) 0 else maximum(policy.value_map[:, states_b[state_index]]) end
            new_value = state_rewards[state_index] + solver.discount * next_value
            new_updated_value = solver.learning_rate * new_value + (1 - solver.learning_rate) * policy.value_map[current_action, current_state]

            policy.value_map[current_action, current_state] = new_updated_value
        end

        last_rewards[i] = sum(state_rewards)

        #if state_reward > 0
        #    println("Epoch: $i, Frames: $(length(state_rewards)), Sum score: $state_reward, Max score: $(maximum(state_rewards)), Mean score: $(round(mean(state_rewards), 2)), Median score: $(median(state_rewards))")
        #end

        if verbose && (i > 100 && i % solver.print_every_n == 0)
            cum_sum_reward = sum(last_rewards[i-101:i])
            cum_avg_reward = mean(last_rewards[i-101:i])
            println("Epoch: $i, cummulative reward: $(round(cum_avg_reward, 2))/$(round(cum_sum_reward, 2))")
        end
    end

    return policy

end
