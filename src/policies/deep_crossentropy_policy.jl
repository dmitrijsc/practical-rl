using StatsBase
using MXNet

import POMDPs: MDP, Policy

#
# Define a policy that will act as a deep crossentropy policy
#
type DeepCrossentropyPolicy <: Policy
    action_space::Vector{Int64}
    net::MXNet.mx.FeedForward
end

#
# Default uniform intializer based on a games input
#
function DeepCrossentropyPolicy(pomdp::MDP; nnet_spec::Vector{Int64} = [24, 48])

    action_count = n_actions(pomdp)
    state_length = n_states(pomdp)

    return DeepCrossentropyPolicy(state_length, action_count, nnet_spec)
end

#
# Model initializer by specifying number of states, actions and neural network parameters
#
function DeepCrossentropyPolicy(state_length::Int64, action_count::Int64, nnet_spec::Vector{Int64}, nnet_optimizer = mx.ADAM())

    action_space = collect(0:(action_count-1))

    # add number of classes as a last hidden layyer if it is different
    # from the definition supplied by user
    if nnet_spec[end] != action_count
        push!(nnet_spec, action_count)
    end

    # create simple mlp model
    mlp = @mx.chain mx.Variable(:data) => mx.MLP(nnet_spec) => mx.SoftmaxOutput(mx.Variable(:label))
    model = mx.FeedForward(mlp)

    # initialize data provider with identical dataset for all possible outcomes
    # this is done to have initial weight initialization
    state_space = repmat(rand(state_length), 1, action_count)

    data_provider = mx.ArrayDataProvider(:data => state_space, :label => action_space)
    mx.fit(model, nnet_optimizer, data_provider, initializer = mx.NormalInitializer(0.0, 0.1), n_epoch = 1, callbacks = [mx.speedometer()])

    DeepCrossentropyPolicy(action_space, model)
end


#
# Select random action from a policy based on a state and action weight
#
function action(policy::DeepCrossentropyPolicy, s::Vector{Float64})

    # we need to reshape our single state into a state vector of size 1
    # so it is supported by mxnet predict function
    current_state_input = reshape(s, (length(s), 1))
    data_provider = mx.ArrayDataProvider(:data => current_state_input)

    probs = mx.predict(policy.net, data_provider; verbosity = 0)[:, 1]

    sample(policy.action_space, Weights(probs))
end
