import torch

class LinearModelClass(torch.nn.Module):

    def __init__(self):

        super(LinearModelClass, self).__init__()

        self.input_size = 256*256*4     # How many input pixels * channels  are in our image?

        self.output_size = 4    # How many different classes are we trying to predict from?

        self.hidden_size = 2048    # Choose any number you'd like for connecting the layers.
                          # A number too small will lead to an information bottleneck.
                          # A number too large is unnecessary and harder to train.
                          # Suggested values: 512, 1024, 2048

        self.model = torch.nn.Sequential( # This stacks multiple layers together to be called at once.

            # We need this in order to read a 2D image and its color channels as a single vector
            torch.nn.Flatten(1),

            # First hidden layer            
            torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_size),

            # Choose an activation function (Use documenation above).
            # torch.nn.______(),
            torch.nn.ReLU(),

            # You may add additional hidden layers if you'd like. 
            # The in features must match the previous layer's out features, 
            #   and the out_features matches next layer's in features.
            # Remember to add activation layers after each hidden layer.



            # Final (output) layer
            torch.nn.Linear(self.hidden_size, self.output_size)

            # Do not add activation layers or regularization after the final layer.
            # This will lead to information loss. PyTorch will add the softmax for us.
        )      

    def forward(self, x):

        out = self.model(x)

        return out
