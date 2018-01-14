include("src/base.jl")
include("src/envs/classiccontrol_mdp.jl")
include("src/policies/deep_crossentropy_policy.jl")
include("src/solvers/deep_crossentropy_policy_solver.jl")

gym_env = ClassicControlMDP(gym.make("CartPole-v0")[:env])
solver = DeepCrossentropyPolicySolver(100, 100, 0.7; max_frame_iterations = 200, print_every_n = 25, nnet_spec = [20, 10])
policy = solve(solver, gym_env, verbose = true)

# print(policy.action_map)
