from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers -
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Output - one for each digit
        self.fc1 = nn.Linear(16 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

        # max pool
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])  # correct view

        # convolutional layers
        x = self.maxpool(self.leaky_relu(self.conv1(x)))
        x = self.maxpool(self.leaky_relu(self.conv2(x)))
        x = self.maxpool(self.leaky_relu(self.conv3(x)))

        # Flatten x to send through fully connected layers
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.log_softmax(self.out(x))
        return x
