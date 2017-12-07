using OpenAIGym
import Reinforce.action

# FrozenLake
# Today you are going to learn how to survive walking over the (virtual) frozen lake through discrete optimization.

# Setup the environment
env = GymEnv("FrozenLake8x8-v0")
env_actions = env.pyenv[:action_space][:n]
env_space = env.pyenv[:observation_space][:n]

#
# Create a new episode and take action "right" (2)
#

# Frozen Lake policy to move right
type FLMoveRightPolicy <: AbstractPolicy end
action(policy::FLMoveRightPolicy, r, s′, A′) = 2

# Initiate episode using FLMoveRightPolicy
ep = Episode(env, FLMoveRightPolicy())

for (s, a, r, sp) in ep
    println("Start state: $s, Last action: $a, Reward: $r, New state: $sp, Complete: $(env.done)")
end

#
# Baseline: random search
# The environment has a 4x4 grid of states (16 total), they are indexed from 0 to 15
# From each states there are 4 actions (left,down,right,up), indexed from 0 to 3
# We need to define agent's policy of picking actions given states. Since we have only
# 16 disttinct states and 4 actions, we can just store the action for each state in an array.
# This basically means that any array of 16 integers from 0 to 3 makes a policy.
#

#
# Lets define a function that will run a single game
#
function episode(env, policy = RandomPolicy(); stepfunc = nothing, kw...)

    reward = run_episode(env, policy) do
        # Nothing here
    end

    return reward
end

# Set number of episodes to play
play_episode_number = 10^3

# Initialize rewards array with 0
rewards = zeros(play_episode_number)

# Run episode for a predefined number of games
map!(x -> episode(env), rewards)

# Evaluate the results
println("Total rewards: $(sum(rewards)), mean value: $(mean(rewards))")
