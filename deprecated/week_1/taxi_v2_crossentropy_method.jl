using OpenAIGym, StatsBase
import Reinforce.action

# Crossentropy method
# This notebook will teach you to solve reinforcement learning with crossentropy method.
env = GymEnv("Taxi-v2") # Taxi-v2 for 4x4
# env.pyenv[:spec][:max_episode_steps]=1000

env_actions = env.pyenv[:action_space][:n]
env_space = env.pyenv[:observation_space][:n]
env_space_actions = 0:(env_space - 1)


# Create stochastic policy
# This time our policy should be a probability distribution.
# policy[a,s] = P(take action a | in state s)
# Since we still use integer state and action representations, you can use a 2-dimensional array to represent the policy.
# Please initialize policy uniformly, that is, probabililities of all actions should be equal.
policy_values = ones(env_actions, env_space) * 1.0 / env_actions

# Define a new policy that will store probability mapping inside the policy
struct StateProbActionPolicy <: AbstractPolicy
    states::Array{Float64,2}
end

# Define the policy action function
function action(policy::StateProbActionPolicy, r, s, A′)

    state_index = s[] + 1 # fixing bug, when using Julia
    action_probs = policy.states[:, state_index]

    sample(env_space_actions, Weights(action_probs))
end

# Play the game
# Just like before, but we also record all states and actions we took.
function play_episode(policy, t_max = 10^3)

    # Initiate return with a fixed size
    states, actions = zeros(t_max), zeros(t_max)
    total_reward = 0.

    # Reset environment and episode data
    reset!(env)
    ep = Episode(env, policy)
    frames_played = 0

    # Play episode until we reach episode end or t_max
    for (s, a, r, sp) in ep

        # println("Start state: $s, Last action: $a, Reward: $r, New state: $sp, Complete: $(env.done) ($frames_played)")

        frames_played += 1
        total_reward += r

        states[frames_played] = sp
        actions[frames_played] = a

        if frames_played >= t_max || env.done
            break
        end
    end

    # We only pick those frames that actually happened
    return states[1:frames_played], actions[1:frames_played], total_reward
end

# Training loop
# Generate sessions, select N best and fit to those.
function train(epochs = 10)

    global policy_values

    n_samples = 500  #sample this many samples
    quantile_threshold = .75  #take this percent of session with highest rewards
    smoothing = .1  #add this thing to all counts for stability
    learning_rate = 0.1

    for i = 1:epochs

        batch_states  = Vector{Int64}[] # a list of lists of states in each session
        batch_actions = Vector{Int64}[] # a list of lists of actions in each session
        batch_rewards = Float64[] # a list of floats - total rewards at each session

        for j=1:n_samples

            if j % 50 == 0
                println("epoch: $i, episode: $j")
            end

            session_states, session_actions, session_reward = play_episode(StateProbActionPolicy(policy_values))
            push!(batch_states, session_states)
            push!(batch_actions, session_actions)
            append!(batch_rewards, session_reward)
        end

        # Identify threshold which filters out best performing games
        reward_threshold = quantile(batch_rewards, quantile_threshold);
        reward_valid_indices = batch_rewards .> reward_threshold;

        # Create a filtered view to best peforming games
        filtered_states  = view(batch_states, reward_valid_indices);
        filtered_actions = view(batch_actions, reward_valid_indices);

        # Create a new policy matrix which has identical structure and smoothing value
        policy_values = zeros(policy_values) + smoothing;
        for ep in 1:sum(reward_valid_indices)
            map(x -> policy_values[filtered_actions[ep][x] + 1, filtered_states[ep][x] + 1] += 1 , 1:length(filtered_actions[ep]));
        end

        policy_values = policy_values ./ sum(policy_values, 1)

        # policy_values_new = zeros(policy_values) + smoothing
        #
        # for ep in 1:sum(reward_valid_indices)
        #     map(x -> policy_values_new[filtered_actions[ep][x] + 1, filtered_states[ep][x] + 1] += 1 , 1:length(filtered_actions[ep]));
        # end
        #
        # # Update policy values probabilities
        # policy_values_new = policy_values_new ./ sum(policy_values_new, 1)
        # policy_values += (learning_rate * policy_values_new)
        # policy_values = policy_values ./ sum(policy_values, 1)

        max_frames = maximum(x -> length(x), batch_actions)
        mean_frames = mean(x -> length(x), batch_actions)

        println("epoch: $i, mean reward: $(mean(batch_rewards)), max reward: $(maximum(batch_rewards)), threshold: $reward_threshold, max_frames: $max_frames $mean_frames")
    end

end

train()
