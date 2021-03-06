import PyCall
import POMDPs: MDP, Policy

#
# Atari environment used to play Open AI Classic Control games: https://gym.openai.com/envs/#classic_control
#
# PS. It also supports multi-dimensional states, but it wont solve them efficiently
# as they will get converted to 1 dimensional array and will be used in a fully
# connected layer.
#
struct ClassicControlMDP <: MDP{Vector{Float64}, Int64}
    env::PyCall.PyObject
    state_space_size::Int64
    action_space::Vector{Int64}
end

#
# Initializer function that is using PyObject with Classic Control environment
# to initialize the variables
#
function ClassicControlMDP(env::PyCall.PyObject)

    states = prod(env[:observation_space][:shape])
    actions = Vector{Int64}(0:(env[:action_space][:n] - 1))

    return ClassicControlMDP(env, states, actions)
end

#
# We define number of function required to play the game
#
states(mdp::ClassicControlMDP) = mdp.state_space_size
actions(mdp::ClassicControlMDP) = mdp.action_space
n_states(mdp::ClassicControlMDP) = mdp.state_space_size
n_actions(mdp::ClassicControlMDP) = length(mdp.action_space)
initial_state(mdp::ClassicControlMDP, rng::AbstractRNG = MersenneTwister(0)) = mdp.env[:reset]()[:] .* 1.0
isterminal(mdp::ClassicControlMDP, s::Vector{Float64}) = iszero(s)
action_index(mdp::ClassicControlMDP, a::Int64) = a

#
# The only function we can use from OpenAIGym is getting a reward when taking a
# particular action in a specific step
#
function generate_sr(mdp::ClassicControlMDP, s::Vector{Float64}, a::Int64)

    state, reward, done = mdp.env[:step](a)

    if done
        return (state[:] .* 0., reward)
    end

    return (state[:] * 1., reward)
end


# Play a single episode
function run_experiment(env::ClassicControlMDP, policy::Policy; max_frame_iterations::Int64 = 100, keep_history = false)

    reward::Float64 = .0
    previous_state::Vector{Float64} = initial_state(env)

    #
    # We might decide to keep the history for some of our learning algorithms
    # therefore we need to initialize lists of a fixed size
    #
    list_size = if keep_history max_frame_iterations else zero(max_frame_iterations) end

    states = zeros(Float64, env.state_space_size, list_size)
    new_states = zeros(Float64, env.state_space_size, list_size)
    actions = zeros(Int64, list_size)
    rewards = zeros(Float64, list_size)

    frames_played = 0

    for i=1:max_frame_iterations

        current_action_i = action(policy, previous_state)
        current_action = action_index(env, current_action_i)
        current_state, current_reward = generate_sr(env, previous_state, current_action)
        reward += current_reward
        frames_played += 1

        #
        # In case we keep history lets track it
        #
        if keep_history
            states[:, i] = previous_state
            new_states[:, i] = current_state
            actions[i] = current_action_i
            rewards[i] = current_reward
        end

        if isterminal(env, current_state)
            break
        end

        previous_state = current_state

    end

    #
    # In case we keep history we should also cut in based on a number of frames required
    #
    if keep_history
        return states[:, 1:frames_played], actions[1:frames_played], reward, new_states[1:frames_played], rewards[1:frames_played]
    else
        return states, actions, reward, new_states
    end
end

#
# Executes policy on an environment for a number of repeats with a
# limitation on how many actions/ frames the episode can handle
#
function execute_policy(env::ClassicControlMDP, policy::Policy, experiment_repeats::Int64, max_frame_iterations::Int64; experiment_index = nothing, verbose = false)

    if experiment_index != nothing && verbose == true
        println("Policy execution #$experiment_index")
    end

    return mean(map(x -> run_experiment(env, policy, max_frame_iterations)[3], zeros(experiment_repeats)))
end
