import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape=(2, 1024), classes=26):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 33), padding='same')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2, 33), padding='same')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2, 33), padding='same')
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(256, 64, kernel_size=(2, 33), padding='same')
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout4 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(64 * input_shape[0] * (input_shape[1] // 16), 128)
        self.relu5 = nn.ReLU()
        self.dense2 = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu5(x)
        x = self.dense2(x)
        # x = self.softmax(x)

        return x


class Conv1dNet(nn.Module):
    def __init__(self, input_shape=(2, 1024), classes=26):
        super(Conv1dNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(2, 64, 5, 3, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, 3, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, 3, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 5, 3, 0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 5, 3, 0),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.frft_feature = nn.Sequential(
            nn.Conv1d(1, 64, 5, 3, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, 3, 0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, 3, 0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 5, 3, 0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 5, 3, 0),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, classes)
        )
    def forward(self, x):
        amp, phs = x
        # x1 = self.feature(amp)
        # x1 = x1.view(x1.shape[0], -1)
        x2 = self.frft_feature(phs)
        x2 = x2.view(x2.shape[0], -1)
        # x = torch.concat((x1, x2), axis=-1)
        x = self.fc(x2)
        return x
