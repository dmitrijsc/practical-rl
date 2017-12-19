import PyCall
import POMDPs: MDP, Solver, Policy

if !isdefined(:gym)
    global const gym = PyCall.pywrap(PyCall.pyimport("gym"))
end




gym_env = AtariGymMDP(gym.make("FrozenLake-v0"))
solver = EvolutionPolicySolver(training_experiment_count = 100, experiment_repeats = 100)
policy = solve(solver, gym_env, verbose = true)

print(policy.action_map)

#n_observations(::AtariGymMDP) = 2;
