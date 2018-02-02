include("src/base.jl")
include("src/envs/toytext_mdp.jl")
include("src/policies/value_table_policy.jl")
include("src/solvers/ev_sarsa_policy_solver.jl")

gym_env = ToyTextMDP(gym.make("Taxi-v2"))
solver = EVSarsaPolicySolver(5000; max_frame_iterations = 100, epsilon = 0.25, epsilon_discount = 0.995, learning_rate = 0.1, discount = 0.99)
policy = solve(solver, gym_env, verbose = true)

# println(policy.value_map)
