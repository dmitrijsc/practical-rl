include("src/base.jl")
include("src/envs/toytext_mdp.jl")
include("src/policies/evolution_policy.jl")
include("src/solvers/evolution_policy_solver.jl")

gym_env = ToyTextMDP(gym.make("FrozenLake-v0"))
solver = EvolutionPolicySolver(training_experiment_count = 500, experiment_repeats = 100)
policy = solve(solver, gym_env, verbose = true)

print(policy.action_map)
