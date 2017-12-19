import POMDPs: Solver

type EvolutionPolicySolver <: Solver
    training_experiment_count::Int64
    experiment_repeats::Int64
    max_frame_iterations::Int64
    epochs::Int64
    imputations::Int64
    keep_records::Int64
    print_every_n::Int64
end

EvolutionPolicySolver(;training_experiment_count::Int64=10^2, experiment_repeats::Int64=10, max_frame_iterations::Int64=100, epochs::Int64=10, imputations::Int64=50, keep_records::Int64=100, print_every_n = 10) =
    EvolutionPolicySolver(training_experiment_count, experiment_repeats, max_frame_iterations, epochs, imputations, keep_records, print_every_n)

function solve(solver::EvolutionPolicySolver, pomdp::MDP; verbose = true)

    # Play a single episode
    function run_experiment(env, policy, max_frame_iterations)

        reward::Float64 = .0

        previous_state::Int64 = initial_state(env)

        for i=1:max_frame_iterations

            current_action = action_index(env, action(policy, previous_state))
            current_state, current_reward = generate_sr(env, previous_state, current_action)
            reward += current_reward

            if isterminal(env, current_state)
                break
            end

            previous_state = current_state

        end

        reward
    end

    #
    # Executes policy on an environment for a number of repeats with a
    # limitation on how many actions/ frames the episode can handle
    #
    function execute_policy(env, policy, experiment_repeats, max_frame_iterations; experiment_index = nothing, verbose = false)

        if experiment_index != nothing && verbose == true
            println("Policy Index: $experiment_index")
        end

        return sum(map(x -> run_experiment(env, policy, max_frame_iterations), zeros(experiment_repeats)))
    end

    policy_values = map(x -> EvolutionPolicy(pomdp), 1:solver.training_experiment_count)
    policy_scores = map(x -> execute_policy(pomdp, policy_values[x], solver.experiment_repeats, solver.max_frame_iterations; experiment_index = x, verbose = x == 1 || x % solver.print_every_n == 0), 1:solver.training_experiment_count)

    #
    # Identify best policy and update score variable accordingly
    #
    for i = 1:solver.epochs

        # Current size can be different, for example if the initial data set is
        # larger than KEEP_RECORDS
        current_size = length(policy_scores)

        # We are imputing 2 set of records as a part of our genetic algorithm
        # 1. Random combination of 2 random policies
        # 2. Random combination of a single random policy from collection and random change
        for j = 1:solver.imputations

            # We are running single random generator to generate policy number For
            # both of the approaches we are gonna use
            # policy_selector = Int.(round.(rand(3, 1) * (CURRENT_SIZE - 1))) + 1
            policy_selector = rand(1:current_size, 3, 1)
            verbose_current = verbose && (j == 1 || j % solver.print_every_n == 0)

            # Debug information on a step info
            if verbose_current
                println(i, " - ", j, " / ", policy_selector[1], " - ", policy_selector[2])
            end

            # 1. Random combination of 2 random policies + appending the results
            policy_values = vcat(policy_values, EvolutionPolicy(pomdp, policy_values[policy_selector[1]], policy_values[policy_selector[2]]))
            policy_scores = vcat(policy_scores, execute_policy(pomdp, policy_values[current_size + j*2 - 1], solver.experiment_repeats, solver.max_frame_iterations))

            # 2. Random combination of a single random policy from collection and random change
            policy_values = vcat(policy_values, EvolutionPolicy(pomdp, policy_values[policy_selector[3]]))
            policy_scores = vcat(policy_scores, execute_policy(pomdp, policy_values[current_size + j*2], solver.experiment_repeats, solver.max_frame_iterations))
        end

        # We are identifying indexes ASC of our dataset based on a policy_scores
        indices = sortperm(policy_scores)

        # We keep only KEEP_RECORDS best performing records
        policy_values = policy_values[indices][end-(solver.keep_records - 1):end]
        policy_scores = policy_scores[indices][end-(solver.keep_records - 1):end]

        # Debug information
        if verbose
            println("Epoch: $i, Max score: $(maximum(policy_scores)), Mean score: $(round(mean(policy_scores), 0)), Median score: $(median(policy_scores))")
        end

    end

    best_policy_index = sortperm(policy_scores)[end]
    best_policy = policy_values[best_policy_index]
    best_policy.score = policy_scores[best_policy_index]

    return best_policy

end
