import PyCall
import POMDPs: MDP, Policy

using Images, Colors


#
# Atari environment is used for playing OpenAI Atari https://gym.openai.com/envs/#atari
# Games are provided in a form of a 3-channel RGB image. 
# P.S. We are taking every second pixel of a frame.
#
# Our environment will consist of Python variable and action space
#
struct AtariMDP <: MDP{Vector{Float64,}, Int64}
    env::PyCall.PyObject
    action_space::Vector{Int64}
end

#
# Classic initializer which is setting action space
#
function AtariMDP(env::PyCall.PyObject)

    actions = Vector{Int64}(0:(env[:action_space][:n] - 1))

    return AtariMDP(env, actions)
end

#
# We define number of function required to play the game
#
states(mdp::AtariMDP) = -1
actions(mdp::AtariMDP) = mdp.action_space
n_states(mdp::AtariMDP) = -1
n_actions(mdp::AtariMDP) = length(mdp.action_space)
initial_state(mdp::AtariMDP, rng::AbstractRNG = MersenneTwister(0)) = state_view(mdp, mdp.env[:reset]())
isterminal(mdp::AtariMDP, s) = iszero(s)
action_index(mdp::AtariMDP, a::Int64) = a

# We take every second pixel, swap the dimensions to be image and 
# convert an image to grayscale
state_view(mdp::AtariMDP, s) = channelview(Gray.(colorview(RGB, permutedims(s[1:2:end, 1:2:end, :] * 1.0, (3, 1, 2)) ./ 255.0)))

#
# The only function we can use from OpenAIGym is getting a reward when taking a
# particular action in a specific step
#
function generate_sr(mdp::AtariMDP, s, a::Int64)

    state, reward, done = mdp.env[:step](a)

    if done
        return (zeros(state_view(mdp, state)), reward)
    end

    return (state_view(mdp, state), reward)
end