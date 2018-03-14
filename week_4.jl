include("src/base.jl")
include("src/envs/classiccontrol_mdp.jl")
include("src/policies/dqn_policy.jl")
include("src/solvers/dqn_policy_solver.jl")

gym_env = ClassicControlMDP(gym.make("CartPole-v0")[:env])
solver = DQNPolicySolver(500; nnet_spec = Int64[4], epsilon = 0.25, learning_rate = 0.00001, discount = 0.1)
policy = solve(solver, gym_env; verbose = true)
