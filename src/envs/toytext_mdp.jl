import PyCall
import POMDPs: MDP

#
# In case you are planning to use the code for anything else than ToyText from
# OpenAI gym please pay attention to `initial_state`, `isterminal` and
# `action_index` functions as they are OpenAI specific. For example, OpenAI
# gym assume indices start from 0, but we change it to 1 etc.
#

#
# Define a new type responsible for working with ToyText from OpenAi gym
# It also supports other environments with a fixes number of states and actions
#
immutable ToyTextMDP <: MDP{Int64, Int64}
    env::PyCall.PyObject
    state_space::Vector{Int64}
    action_space::Vector{Int64}
end

#
# Initializer function that is using PyObject with ToyText environment
# to initialize the variables
#
function ToyTextMDP(env::PyCall.PyObject)

    states = Vector{Int64}(1:env[:observation_space][:n])
    actions = Vector{Int64}(1:env[:action_space][:n])

    return ToyTextMDP(env, states, actions)
end

#
# We define number of function required to play the game
#
states(mdp::ToyTextMDP) = mdp.state_space
actions(mdp::ToyTextMDP) = mdp.action_space
n_states(mdp::ToyTextMDP) = length(mdp.state_space)
n_actions(mdp::ToyTextMDP) = length(mdp.action_space)
initial_state(mdp::ToyTextMDP, rng::AbstractRNG = MersenneTwister(0)) = mdp.env[:reset]()[] + 1
isterminal(mdp::ToyTextMDP, s::Int64) = s == 0
action_index(mdp::ToyTextMDP, a::Int64) = a - 1

#
# The only function we can use from OpenAIGym is getting a reward when taking a
# particular action in a specific step
#
function generate_sr(mdp::ToyTextMDP, s::Int64, a::Int64, rng::AbstractRNG)

    state, reward, done = mdp.env[:step](a)

    if done
        return (zero(state), reward)
    end

    return (state + 1, reward)
end
