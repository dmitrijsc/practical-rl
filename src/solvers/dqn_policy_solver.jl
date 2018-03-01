import POMDPs: MDP, Solver

mutable struct DQNPolicySolver <: Solver
    epochs::Int64
    nnet_spec::Vector{Int64}
    epsilon::Float64
    epsilon_discount::Float64
    learning_rate::Float64
    discount::Float64
    max_frame_iterations::Int64
    print_every_n::Int64
end

DQNPolicySolver(epochs; nnet_spec::Vector{Int64} = [10, 5], epsilon = 0.5, epsilon_discount = 0.999, learning_rate = 0.005, discount = 0.99, max_frame_iterations = 100, print_every_n = 25) =
    DQNPolicySolver(epochs, nnet_spec, epsilon, epsilon_discount, learning_rate, discount, max_frame_iterations, print_every_n)

function solve(solver::DQNPolicySolver, pomdp::MDP; verbose = true)

    policy = DQNPolicy(pomdp; nnet_spec = solver.nnet_spec, nnet_optimizer = mx.ADAM(lr = solver.learning_rate), epsilon = solver.epsilon, epsilon_discount = solver.epsilon_discount)
    last_rewards = zeros(Float64, solver.epochs)

    for i = 1:solver.epochs

        if verbose && (i == 1 || i % solver.print_every_n == 0)
            println("Epoch: $i, Epsilon: $(round(policy.epsilon, 2))")
        end

        #
        # Q-learning implementation that updates on every step
        #
        previous_state::Vector{Float64} = initial_state(pomdp)
        rewards = 0.0

        for j=1:solver.max_frame_iterations

            current_action_i, current_values = action(policy, previous_state)
            current_action = action_index(pomdp, current_action_i)
            current_state, current_reward = generate_sr(pomdp, previous_state, current_action)
            current_state_terminal = isterminal(pomdp, current_state)

            next_value = if current_state_terminal 0 else maximum(values(policy, current_state)) end
            new_value = current_reward + solver.discount * next_value
            current_values[current_action_i + 1] = new_value

            update(policy, previous_state, current_values)
            
            previous_state = current_state
            rewards += current_reward

            if current_state_terminal
                break
            end
        end

        if 0 < solver.epsilon_discount < 1
            policy.epsilon *= solver.epsilon_discount
        end

        # println(rewards)
        last_rewards[i] = rewards

        if verbose && (i > 100 && i % solver.print_every_n == 0)
            cum_sum_reward = sum(last_rewards[i-101:i])
            cum_avg_reward = mean(last_rewards[i-101:i])
            println("Epoch: $i, Cummulative reward: $(round(cum_avg_reward, 2))/$(round(cum_sum_reward, 2)), Epsilon: $(policy.epsilon)")
        end
    end

    return policy

end
