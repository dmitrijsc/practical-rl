import POMDPs: MDP, Solver

type EVSarsaPolicySolver <: Solver
    epochs::Int64
    epsilon::Float64
    epsilon_discount::Float64
    learning_rate::Float64
    discount::Float64
    max_frame_iterations::Int64
    print_every_n::Int64
end

EVSarsaPolicySolver(epochs; epsilon = 0.25, epsilon_discount = 0.99, learning_rate = 0.8, discount = 0.8, max_frame_iterations = 100, print_every_n = 25) =
    EVSarsaPolicySolver(epochs, epsilon, epsilon_discount, learning_rate, discount, max_frame_iterations, print_every_n)

function solve(solver::EVSarsaPolicySolver, pomdp::MDP; verbose = true)

    policy = ValueTablePolicy(pomdp, solver.epsilon, solver.learning_rate, solver.discount)
    last_rewards = zeros(Float64, solver.epochs)
    actions_count = n_actions(pomdp)

    for i=1:solver.epochs

        if verbose && (i == 1 || i % solver.print_every_n == 0)
            println("Epoch: $i")
        end

        #
        # SARSA implementation that updates on every step
        #
        previous_state::Int64 = initial_state(pomdp)
        rewards = 0.0

        for j=1:solver.max_frame_iterations

            current_action_i = action(policy, previous_state)
            current_action = action_index(pomdp, current_action_i)
            current_state, current_reward = generate_sr(pomdp, previous_state, current_action)
            current_state_terminal = isterminal(pomdp, current_state)

            # expected value sarsa sum all values from a next state
            next_value = if current_state_terminal 0 else  sum(policy.value_map[:, current_state]) end
            new_value = current_reward + solver.discount * next_value / actions_count
            new_updated_value = solver.learning_rate * new_value + (1 - solver.learning_rate) * policy.value_map[current_action_i, previous_state]

            policy.value_map[current_action_i, previous_state] = new_updated_value
            previous_state = current_state
            rewards += current_reward

            if current_state_terminal
                break
            else

            end
        end

        if 0 < solver.epsilon_discount < 1
            policy.epsilon *= solver.epsilon_discount
        end

        last_rewards[i] = rewards

        if verbose && (i > 100 && i % solver.print_every_n == 0)
            cum_sum_reward = sum(last_rewards[i-101:i])
            cum_avg_reward = mean(last_rewards[i-101:i])
            println("Epoch: $i, Cummulative reward: $(round(cum_avg_reward, 2))/$(round(cum_sum_reward, 2)), Epsilon: $(policy.epsilon)")
        end
    end

    return policy

end
