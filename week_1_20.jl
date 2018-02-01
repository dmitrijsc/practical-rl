
using MXNet
using Distributions
#using Plots

# create a two hidden layer MPL: try varying num_hidden, and change tanh to relu,
# or add/remove a layer
data = mx.Variable(:data)
label = mx.Variable(:label)
net = @mx.chain     mx.Variable(:data) =>
                    mx.MLP([5, 5, 2])            =>
                    mx.SoftmaxOutput(mx.Variable(:label))

# final model definition, don't change, except if using gpu
model = mx.FeedForward(net, context=mx.cpu())

# set up the optimizer: select one, explore parameters, if desired
#optimizer = mx.SGD(lr=0.01, momentum=0.9, weight_decay=0.00001)
optimizer = mx.ADAM(lr=0.01)


trainprovider = mx.ArrayDataProvider(:data => [[0,0,0] [1,1,1]], :label => [1, 1])
trainprovider = mx.ArrayDataProvider(:data => [[0,0,0] [0,0,0]], :label => [1, 1])
trainprovider = mx.ArrayDataProvider(:data => [[1,2,2,2, 5] [1,2,3,4,4] [1,0, 0,0, 0]], :label => [1, 1, 0])
evalprovider = mx.ArrayDataProvider(:data => [[1,2,4,4, 5] [1,2,3,4,4] [0,0, 0,0, 0]], :label => [1, 1, 0])
testprovider = mx.ArrayDataProvider([[1,1,1] [0,0,0]])

# train, reporting loss for training and evaluation sets
# initial training with small batch size, to get to a good neighborhood
mx.fit(model, optimizer, trainprovider,
       initializer = mx.NormalInitializer(0.0, 0.1),
       #eval_metric = mx.MSE(),
       # eval_data = evalprovider,
       n_epoch = 10,
       callbacks = [mx.speedometer()])

mx.predict(model, testprovider)

trainprovider, evalprovider = data_source(#= batchsize =# samplesize)
mx.fit(model, optimizer, trainprovider,
       initializer = mx.NormalInitializer(0.0, 0.1),
       eval_metric = mx.MSE(),
       eval_data = evalprovider,
       n_epoch = 500,  # previous setting is batchsize = 200, epoch = 20
                       # implies we did (5000 / 200) * 20 times update in previous `fit`
       callbacks = [mx.speedometer()])
