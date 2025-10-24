import torch.nn as nn

class HeavyNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool1d(kernel_size=10, stride=10),
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.output_channels = 128
        # Calculation for output length:
        # Input size: 1250 -> After MaxPool1d(10): 125
        # Conv1 (stride 1, padding 1): 125
        # Conv2 (stride 2, padding 1): (125 - 3 + 2*1)/2 + 1 = 63
        # Conv3 (stride 1, padding 1): 63
        # Conv4 (stride 2, padding 1): (63 - 3 + 2*1)/2 + 1 = 32
        # Conv5 (stride 1, padding 1): 32
        # Conv6 (stride 2, padding 1): (32 - 3 + 2*1)/2 + 1 = 16
        # Conv7 (stride 1, padding 1): 16
        # Conv8 (stride 2, padding 1): (16 - 3 + 2*1)/2 + 1 = 8
        self.output_length = 8

    def forward(self, x):
        return self.features(x)

class HeavyNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = HeavyNetFeatureExtractor()
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