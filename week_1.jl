include("src/base.jl")
include("src/envs/toytext_mdp.jl")
include("src/policies/crossentropy_policy.jl")
include("src/solvers/crossentropy_policy_solver.jl")

#gym_env = ToyTextMDP(gym.make("Taxi-v2"))
gym_env = ToyTextMDP(gym.make("FrozenLake8x8-v0"))

solver = CrossentropyPolicySolver(100, 500, 0.75, 0.1; max_frame_iterations = 1000, print_every_n = 100, min_score = 0.0, min_score_step = 0.0001)
policy = solve(solver, gym_env, verbose = true)

println(policy.action_map)
