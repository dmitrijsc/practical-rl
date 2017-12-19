using OpenAIGym
import Reinforce.action

# FrozenLake
# Today you are going to learn how to survive walking over the (virtual) frozen lake through discrete optimization.

# Setup the environment and retrieve base details
env = GymEnv("FrozenLake8x8-v0") # FrozenLake-v0 for 4x4
env_actions = env.pyenv[:action_space][:n]
env_space = env.pyenv[:observation_space][:n]
env_space_actions = 0:(env_space - 1)

#
# Lets define a new policy function that wont rely on a global variables
# Our new policy will require to pass steps during initialization
#

# Define the policy type with potential states-steps
struct StateActionPolicy <: AbstractPolicy
    states::Vector{Int}
end

# Define the policy action function
function action(policy::StateActionPolicy, r, s′, A′)
    index = (typeof(s′) != Int64 ? 0 : s′) + 1 # there is an issue that initial position is not given
    policy.states[index]
end

#
# Lets define standard parameters, such as:
# 1. Number of unique episodes we play
# 2. Reset their rewards to 0
# 3. Create random action for each episode state
#

# Set number of episodes to play
play_episode_number = 10^3

# Initialize rewards array with 0
rewards = zeros(play_episode_number)

# Define steps for each policy
# random_policy_values = Int.(floor.(rand(env_space, play_episode_number) * (env_actions)))
random_policy_values = rand(env_space_actions, env_space, play_episode_number)

# Play a single episode
function global_episode(env, policy)

    reward = run_episode(env, policy) do
        # Nothing here
    end

    return reward

end

# Plays policy multiple times and aggregates the result
function play_global_policy(env, policy_id = 1, episode_count = 100)
    return sum(map(x -> global_episode(env, StateActionPolicy(random_policy_values[:, policy_id])), zeros(episode_count)))
end

# Runs a different policy for each game
policy_scores = map(x -> play_global_policy(env, x[1]), enumerate(rewards));

# Print out reward results
println(policy_scores)

#
# Part II Genetic algorithm (4 points)
# The next task is to devise some more effecient way to perform policy search.
# We'll do that with a bare-bones evolutionary algorithm.
#

# For each state, with probability p take action from policy1, else policy2
function crossover(policy_1_id, policy_2_id, p = 0.5)

    l = env_space
    m = ifelse.(rand(l) .< p, zeros(l), ones(l))

    m .* random_policy_values[:, policy_1_id] + (1 - m) .* random_policy_values[:, policy_2_id]

end

# For each state, with probability p replace action with random action
function mutate(policy_1_id, p = 0.1)

    l = env_space
    m = ifelse.(rand(l) .< p, zeros(l), ones(l))
    r = rand(env_space_actions)

    m .* random_policy_values[:, policy_1_id] + (1 - m) .* r

end

EPOCHS = 20
IMPUTATIONS = 50
KEEP_RECORDS = 100

for i = 1:EPOCHS

    # Current size can be different, for example if the initial data set is
    # larger than KEEP_RECORDS
    CURRENT_SIZE = length(policy_scores)

    # We are imputing 2 set of records as a part of our genetic algorithm
    # 1. Random combination of 2 random policies
    # 2. Random combination of a single random policy from collection and random change
    for j = 1:IMPUTATIONS

        # We are running single random generator to generate policy number For
        # both of the approaches we are gonna use
        # policy_selector = Int.(round.(rand(3, 1) * (CURRENT_SIZE - 1))) + 1
        policy_selector = rand(1:CURRENT_SIZE, 3, 1)

        # Debug information on a step info
        if j == 1 || j % 10 == 0
            println(i, " - ", j, " / ", policy_selector[1], " - ", policy_selector[2])
        end

        # 1. Random combination of 2 random policies + appending the results
        random_policy_values = hcat(random_policy_values, crossover(policy_selector[1], policy_selector[2]))
        policy_scores = vcat(policy_scores, play_global_policy(env, CURRENT_SIZE + j*2 - 1))

        # 2. Random combination of a single random policy from collection and random change
        random_policy_values = hcat(random_policy_values, mutate(policy_selector[3]))
        policy_scores = vcat(policy_scores, play_global_policy(env, CURRENT_SIZE + j*2))
    end

    # We are identifying indexes ASC of our dataset based on a policy_scores
    indices = sortperm(policy_scores)

    # We keep only KEEP_RECORDS best performing records
    random_policy_values = random_policy_values[:, indices][:, end-KEEP_RECORDS:end]
    policy_scores = policy_scores[indices][end-(KEEP_RECORDS - 1):end]

    # Debug information
    println("\t", maximum(policy_scores), " - ", round(mean(policy_scores), 0), " - ", median(policy_scores))

end
