import torch
from Constants import sliceSize

# Create the network used to predict the up or down for the stock based on event embeddings
# The output should only be a single digit from the output model -- +1 or 0
class DeepPredictionNetwork(torch.nn.Module):

    # Stride is whether or not the convolution jumps across events -- want to cover all the events so 1 will do
    STRIDE = None
    # Padding to be added around the matrix -- No padding since this is narrow convolution
    PADDING = 0
    # Dilation is the spacing between the points used for the convolution -- No need to use this
    DILATION = 0
    # Groups is in / out channels -- not sure but probably 1 is fine
    CONV_GROUPS = 1
    # Kernel size for convolution layer
    CONV_KERNEL_SIZE = 3

    def __init__(self):

        # Call the base constructor
        super(DeepPredictionNetwork, self).__init__()

        # All of the embeddings should have the same number of nodes in their Tensors! Size is k = slice size!
        k = sliceSize

        # Feature model for the long and mid term event embeddings -- Might need separate networks for each due to size of input (30 vs 7)
        self.LTFeatureModel = torch.nn.Sequential(
            # Convolution layer -- l = 3 -- should be convolution over 2d "image" (matrix of event embeddings)
            torch.nn.Conv1d(in_channels=30, # 30 days
                            out_channels=,
                            kernel_size=DeepPredictionNetwork.CONV_KERNEL_SIZE,
                            stride=DeepPredictionNetwork.STRIDE,
                            padding=DeepPredictionNetwork.PADDING,
                            dilation=DeepPredictionNetwork.DILATION,
                            groups=DeepPredictionNetwork.CONV_GROUPS,
                            bias=False),

            # Max pooling layer to retrieve the feature matrix -- make sure the kernel size is correct for the pooling layer
            torch.nn.MaxPool1d(kernel_size=,
                               stride=DeepPredictionNetwork.STRIDE,
                               padding=DeepPredictionNetwork.PADDING,
                               dilation=DeepPredictionNetwork.DILATION,
                               return_indices=False,
                               ceil_mode=False)
        )

        self.MTFeatureModel = torch.nn.Sequential(
            # Convolution layer -- l = 3 -- should be convolution over 2d "image" (matrix of event embeddings)
            torch.nn.Conv1d(in_channels=7, # 7 days
                            out_channels=,
                            kernel_size=DeepPredictionNetwork.CONV_KERNEL_SIZE,
                            stride=DeepPredictionNetwork.STRIDE,
                            padding=DeepPredictionNetwork.PADDING,
                            dilation=DeepPredictionNetwork.DILATION,
                            groups=DeepPredictionNetwork.CONV_GROUPS,
                            bias=False),

            # Max pooling layer to retrieve the feature matrix -- make sure the kernel size is correct for the pooling layer
            torch.nn.MaxPool1d(kernel_size=,
                               stride=DeepPredictionNetwork.STRIDE,
                               padding=DeepPredictionNetwork.PADDING,
                               dilation=DeepPredictionNetwork.DILATION,
                               return_indices=False,
                               ceil_mode=False)
        )

        # Hidden and output layers to be applied to feature vector (Vlt, Vmt, Vst)
        self.outputModel = torch.nn.Sequential(
            # Apply a linear layer for the weights before the first sigmoid -- input is 3 feature vectors length -- neurons in hidden layer are average of input and output layers
            torch.nn.Linear(in_features=3 * k, out_features=((3 * k + 1) / 2), bias=False),
            # Apply the first sigmoid function to get output of hidden layer
            torch.nn.Sigmoid(),
            # Apply a linear layer for the weights before the output layer -- Only one output on the output layer
            torch.nn.Linear(in_features=((3 * k + 1) / 2), out_features=1, bias=False),
            # Apply the final sigmoid function
            torch.nn.Sigmoid()
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
        featureVector = torch.cat((V_lt, V_mt, V_st), 0)

        # Now run the output model
        return self.outputModel(featureVector)