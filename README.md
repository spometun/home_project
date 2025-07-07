### Personal research project on AI basics, using CIFAR10 as a toy dataset. 

## V1. Dynamically killing uneccesary neurons
Implemented per-single-neuron dynamic graph with capability of death of useless neurons, dynamic birth was also in mind, but never implemented.
Neurons where brought to "useless" state by imposing reasonable L1 regularization. Those which started to jump around zero, eventially die in course of training.
With this approach, I was able to obtain 2-layer perceptron, whith 40 neurons in total, which classifies CIFAR10 with 30% accuracy (todo: run again and check exact numbers)

After a while I decided that implementing larger convolutional classifier using per-single-neuron graph is not practical and moved to V2. 

## V2. Smart custom convolutional layers (work in progress)
Goal: Classify CIFAR10 dataset within as small amount of operations as possible.
Operations are counted as single scalar multiply-add, as they would run on general purpose CPU, without GMM structure.
To achieve this, two ideas are employed:
- Death of unused connections form V1 (implemented as masks of custom conv layer)
- Dynamic computational complexity of network (for simpler parts of the image, e.g. big patch of blue sky, it would use less amount of operations)
For this I started to implement custom conv layer capable of doing self-observation and dynamic masking during its work.
Dynamic computational complexity would be efficiently achieved, as RELU outputs a often zeros, and for no further processing required at next layer for such outputs.
Given my custom conv layers a smart, they aim to L1-penealize actual operations, not weights. 
