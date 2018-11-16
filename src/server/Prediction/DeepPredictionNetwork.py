import torch
from math import floor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.interactive(False)

# Import global constants
from Constants import SLICE_SIZE, LT_DAYS, MT_DAYS, ITERATION_NUM

# Import custom layers
from PrintLayer import PrintLayer
from TransformLayer import TransformLayer
from NormalizationLayer import NormalizationLayer

# Create the network used to predict the up or down for the stock based on event embeddings
# The output should only be a single digit from the output model -- +1 or 0
class DeepPredictionNetwork(torch.nn.Module):

    # Stride is whether or not the convolution jumps across events -- want to cover all the events so 1 will do
    STRIDE = 1
    # Padding for the pooling is zero
    POOL_PADDING = 0
    # Padding to be added around the matrix -- No padding since this is narrow convolution
    CONV_PADDING = 1
    # Dilation is the spacing between the points used for the convolution -- No need to use this
    DILATION = 1
    # Groups is in / out channels -- not sure but probably 1 is fine
    CONV_GROUPS = 1
    # Kernel size for convolution layer
    CONV_KERNEL_SIZE = 3

    def __init__(self):

        # Call the base constructor
        super(DeepPredictionNetwork, self).__init__()

        # Initialize the training data
        self.trained = False

        # Size of the input to the feed-forward NN layers
        inputLayer = floor(3 * SLICE_SIZE)
        hiddenLayer = floor((3 * SLICE_SIZE + 1) / 2)

        # Feature model for the long and mid term event embeddings -- Might need separate networks for each due to size of input (30 vs 7)
        self.LTFeatureModel = torch.nn.Sequential(
            # Convolution layer -- l = 3 -- should be convolution over 2d "image" (matrix of event embeddings)
            torch.nn.Conv1d(in_channels=LT_DAYS, # 30 days
                            out_channels=SLICE_SIZE, # k out channels -- Not sure if this is correct -- Turns input from (30, k) -> (k, k)
                            kernel_size=DeepPredictionNetwork.CONV_KERNEL_SIZE,
                            stride=DeepPredictionNetwork.STRIDE,
                            padding=DeepPredictionNetwork.CONV_PADDING,
                            dilation=DeepPredictionNetwork.DILATION,
                            groups=DeepPredictionNetwork.CONV_GROUPS,
                            bias=False),

            # Max pooling layer to retrieve the feature matrix -- make sure the kernel size is correct for the pooling layer -- This has to output 1d
            torch.nn.MaxPool1d(kernel_size=SLICE_SIZE, # Set Kernel size to the length of a row of the output from the conv1d
                               stride=DeepPredictionNetwork.STRIDE,
                               padding=DeepPredictionNetwork.POOL_PADDING,
                               dilation=DeepPredictionNetwork.DILATION,
                               return_indices=False,
                               ceil_mode=False),
            # May have to create flattening layer her to flatten the tensor
            TransformLayer((1, 1, SLICE_SIZE))
        )

        self.MTFeatureModel = torch.nn.Sequential(
            # Convolution layer -- l = 3 -- should be convolution over 2d "image" (matrix of event embeddings)
            torch.nn.Conv1d(in_channels=MT_DAYS, # 7 days
                            out_channels=SLICE_SIZE, # k out channels -- Not sure if this is correct
                            kernel_size=DeepPredictionNetwork.CONV_KERNEL_SIZE,
                            stride=DeepPredictionNetwork.STRIDE,
                            padding=DeepPredictionNetwork.CONV_PADDING,
                            dilation=DeepPredictionNetwork.DILATION,
                            groups=DeepPredictionNetwork.CONV_GROUPS,
                            bias=False),

            # Max pooling layer to retrieve the feature matrix -- make sure the kernel size is correct for the pooling layer
            torch.nn.MaxPool1d(kernel_size=SLICE_SIZE, # Set kernel size to reduce to only k squares -- works as long as k is perfect square
                               stride=DeepPredictionNetwork.STRIDE,
                               padding=DeepPredictionNetwork.POOL_PADDING,
                               dilation=DeepPredictionNetwork.DILATION,
                               return_indices=False,
                               ceil_mode=False),
            # May have to create flattening layer here to flatten the tensor
            TransformLayer((1, 1, SLICE_SIZE))
        )

        # Hidden and output layers to be applied to feature vector (Vlt, Vmt, Vst) -- Not sure if this is correct
        self.outputModel = torch.nn.Sequential(
            # Apply a linear layer for the weights before the first sigmoid -- input is 3 feature vectors length -- neurons in hidden layer are average of input and output layers
            torch.nn.Linear(in_features=inputLayer, out_features=hiddenLayer, bias=False),
            # Apply the first sigmoid function to get output of hidden layer -- This sucks maybe try ReLU again?
            torch.nn.Sigmoid(),
            # Apply a linear layer for the weights before the output layer -- Only two output on the output layer -- -1 and +1 class
            torch.nn.Linear(in_features=hiddenLayer, out_features=2, bias=True),
            # Apply the final sigmoid function
            torch.nn.Sigmoid()
            #torch.nn.Softmax(dim=2)
        )

    # Forward pass of the deep prediction network -- should produce whether stock price increases or decreases
    def forward(self, embeddings):

        # Extract the different embedding periods
        LTEmbeddings, MTEmbeddings, STEmbeddings = embeddings # 30 LTs, 7 MTs, 1 ST (days in period)

        # Get the feature vectors for each time period
        V_lt = self.LTFeatureModel(LTEmbeddings)
        V_mt = self.MTFeatureModel(MTEmbeddings)
        V_st = STEmbeddings

        # Concat the results of the feature vector operations
        featureVector = torch.cat((V_lt, V_mt, V_st), 2)

        # Now run the output model
        return self.outputModel(featureVector)

    # Method to train this network -- Calculate loss and update using standard backpropagation
    def trainNetwork(self, trainingData, epochs=10):

        # Place into training mode
        self.train()

        # Create the loss function and optimizer
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-4, weight_decay=1e-4)  # Set up the optimizer using defaults on Adam (recommended for deep nets)

        lossValues = []
        for _ in range(epochs):

            # Go through the data samples
            for input, target in trainingData:
                # Forward pass
                output = self.forward(input)

                # Compute the loss -- Print the loss to console
                loss = loss_fn(output, target)
                print('Prediction Network: ')
                print(input, output, loss.item())

                # Backward pass -- Zero the gradient to avoid accumulation during backward pass
                optimizer.zero_grad()
                loss.backward()
                lossValues.append(loss.data.numpy())

                # Update the parameters of the optimization function
                optimizer.step()

        self.trained = True
        self.plotLoss(lossValues)
        self.eval()
        # Return the network after training
        return self

    def plotLoss(self, lossValues):

        # Plot the loss of this run
        plt.plot(lossValues)


# Main for temporary testing
if __name__ == '__main__':

    net = DeepPredictionNetwork()

    # net.trainNetwork()

    for x in range(100):
        lt = torch.randn(1, 30, SLICE_SIZE) # These are all correct for the setup -- but maxpooling or conv1d layers causing 1, 256, 239 output shape
        mt = torch.randn(1, 7, SLICE_SIZE)
        st = torch.randn(1, 1, SLICE_SIZE)
        forwardPass = net.forward((lt, mt, st))
        print(forwardPass)