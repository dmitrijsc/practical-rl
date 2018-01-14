include("src/base.jl")
include("src/envs/toytext_mdp.jl")
include("src/policies/value_table_policy.jl")
include("src/solvers/value_table_policy_solver.jl")

gym_env = ToyTextMDP(gym.make("Taxi-v2"))
#gym_env = ToyTextMDP(gym.make("FrozenLake-v0"))

solver = ValueTablePolicySolver(5000; max_frame_iterations = 250, epsilon = 0.01, learning_rate = 0.85, discount = 0.99)
policy = solve(solver, gym_env, verbose = true)

# println(policy.value_map)
