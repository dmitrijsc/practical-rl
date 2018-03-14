using StatsBase
using MXNet

import POMDPs: MDP, Policy

#
# Define a policy that will act as a deep crossentropy policy
#
mutable struct DQNPolicy <: Policy
    action_space::Vector{Int64}
    net::MXNet.mx.FeedForward
    optimizer
    epsilon::Float64
    epsilon_discount::Float64
end

#
# Default uniform intializer based on a games input
#
function DQNPolicy(pomdp::MDP; nnet_spec::Vector{Int64} = [48, 24], nnet_optimizer = mx.ADAM(), epsilon = 0.5, epsilon_discount = 0.999)

    action_count = n_actions(pomdp)
    state_length = n_states(pomdp)

    return DQNPolicy(state_length, action_count, nnet_spec, nnet_optimizer, epsilon, epsilon_discount)
end

#
# Model initializer by specifying number of states, actions and neural network parameters
#
function DQNPolicy(state_length::Int64, action_count::Int64, nnet_spec::Vector{Int64}, nnet_optimizer, epsilon::Float64, epsilon_discount::Float64)

    action_space = collect(0:(action_count-1))

    mlp = @mx.chain mx.Variable(:data)
    for layer_size in nnet_spec
        mlp = mx.FullyConnected(mlp, num_hidden=layer_size)
        mlp = mx.Activation(mlp, act_type=:relu)
    end

    mlp = mx.FullyConnected(mlp, num_hidden=action_count)
    mlp = mx.LinearRegressionOutput(mlp, mx.Variable(:label))

    model = mx.FeedForward(mlp, context = mx.gpu())

    # initialize data provider with identical dataset for all possible outcomes
    # this is done to have initial weight initialization
    random_examples = action_count
    data_provider = mx.ArrayDataProvider(:data => zeros(state_length, random_examples), :label => zeros(action_count, random_examples))
    mx.fit(model, nnet_optimizer, data_provider, initializer = mx.UniformInitializer(0.1), n_epoch = 1, callbacks = [mx.speedometer()])

    DQNPolicy(action_space, model, nnet_optimizer, epsilon, epsilon_discount)
end


#
# Select random action from a policy based on a state and action weight
#
function action(policy::DQNPolicy, s::Vector{Float64})

    # we need to reshape our single state into a state vector of size 1
    # so it is supported by mxnet predict function
    current_values = values(policy, s)
    value, index = findmax(current_values)

    selected_action = if policy.epsilon < rand() index - 1 else rand(policy.action_space) end
    selected_action, current_values
end

function values(policy::DQNPolicy, s::Vector{Float64})
    data_provider = mx.ArrayDataProvider(:data => reshape(s, (length(s), 1)))
    return mx.predict(policy.net, data_provider; verbosity = 0)[:, 1]
end

function update(policy::DQNPolicy, s::Vector{Float64}, v::Vector{Float32})
    data_provider = mx.ArrayDataProvider(:data => repmat(s, 1, length(policy.action_space)), :label => repmat(v, 1, length(policy.action_space)))
    return mx.fit(policy.net, policy.optimizer, data_provider; n_epoch = 3, verbosity = 0)
end