using StatsBase
using MXNet

import POMDPs: MDP, Policy

#
# Define a policy that will act as a deep crossentropy policy
#
type DQNPolicy <: Policy
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

    # add number of classes as a last hidden layyer if it is different
    # from the definition supplied by user
    if length(nnet_spec) > 1
        throw(ArgumentError("nnet_spec: multi layers networks are not yet supported."))
    end

    if nnet_spec[end] != action_count
        push!(nnet_spec, action_count)
    end

    # create simple mlp model
    # nnet_spec = [(3, :tanh), 2]
    # mlp = @mx.chain mx.Variable(:data) => mx.MLP(nnet_spec) => mx.LinearRegressionOutput(mx.Variable(:label))

    mlp = @mx.chain mx.Variable(:data) =>
            mx.FullyConnected(num_hidden=nnet_spec[1]) =>
            mx.Activation(act_type=:tanh) =>
            mx.FullyConnected(num_hidden=action_count) =>
            mx.LinearRegressionOutput(mx.Variable(:label))

    model = mx.FeedForward(mlp)

    # println(model)

    # initialize data provider with identical dataset for all possible outcomes
    # this is done to have initial weight initialization
    random_examples = action_count
    
    # data_provider = mx.ArrayDataProvider(:data => state_space, :label => action_space)
    data_provider = mx.ArrayDataProvider(:data => zeros(state_length, random_examples), :label => zeros(action_count, random_examples))
    mx.fit(model, nnet_optimizer, data_provider, initializer = mx.NormalInitializer(0.0, 0.1), n_epoch = 1, callbacks = [mx.speedometer()])

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
    data_provider = mx.ArrayDataProvider(:data => repmat(s, 1, 2), :label => repmat(v, 1, 2))
    return mx.fit(policy.net, policy.optimizer, data_provider; n_epoch = 3, verbosity = 0)
end