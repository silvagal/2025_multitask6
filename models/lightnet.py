import torch.nn as nn

class LightNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool1d(kernel_size=10, stride=10),
            nn.Conv1d(1, 2, kernel_size=3, stride=2), nn.BatchNorm1d(2), nn.ReLU(),
            nn.Conv1d(2, 4, kernel_size=3, stride=2), nn.BatchNorm1d(4), nn.ReLU(),
        )
        self.output_channels = 4
        # Calculation for output length:
        # Input size: 1250 -> After MaxPool1d(10): 125
        # Conv1: (125 - 3)/2 + 1 = 62
        # Conv2: (62 - 3)/2 + 1 = 30.5 -> 30 (floor)
        self.output_length = 30

    def forward(self, x):
        return self.features(x)

class LightNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = LightNetFeatureExtractor()
        flat_feature_dim = self.feature_extractor.output_channels * self.feature_extractor.output_length

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)