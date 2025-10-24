import torch.nn as nn

class VANetOriginal(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = VANetFeatureExtractor()
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

class VANetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool1d(kernel_size=10, stride=10),
            nn.Conv1d(1, 3, kernel_size=3, stride=2), nn.BatchNorm1d(3), nn.ReLU(),
            nn.Conv1d(3, 2, kernel_size=3, stride=2), nn.BatchNorm1d(2), nn.ReLU(),
            nn.Conv1d(2, 2, kernel_size=3, stride=2), nn.BatchNorm1d(2), nn.ReLU(),
            nn.Conv1d(2, 8, kernel_size=3, stride=2), nn.BatchNorm1d(8), nn.ReLU(),
        )
        self.output_channels = 8
        # A saída desta sequência convolucional será de tamanho (batch, 8, X)
        # O tamanho X depende do SIZE original. Após MaxPool1d(10): SIZE/10 = 125
        # Conv1: (125 - 3)/2 + 1 = 61 + 1 = 62
        # Conv2: (62 - 3)/2 + 1 = 29.5 + 1 = 30 (floor)
        # Conv3: (30 - 3)/2 + 1 = 13.5 + 1 = 14 (floor)
        # Conv4: (14 - 3)/2 + 1 = 5.5 + 1 = 6 (floor)
        self.output_length = 6 # Ajustado com base no cálculo acima
    def forward(self, x):
        return self.features(x)