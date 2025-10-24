import torch
import torch.nn as nn

class HydraNet(nn.Module):
    def __init__(self, feature_extractor_instance, with_pretext_task=False):
        super().__init__()
        self.feature_extractor = feature_extractor_instance
        self.with_pretext_task = with_pretext_task

        # --- Shared Feature Dimension ---
        # The dimension of the feature vector after being flattened.
        flat_feature_dim = self.feature_extractor.output_channels * self.feature_extractor.output_length

        # --- Task-Specific Towers ---
        # Each tower refines the shared features for its specific task.

        # Tower for Main Task (Arrhythmia Classification)
        self.main_tower = nn.Sequential(
            nn.Linear(flat_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Tower for RR Interval Regression
        self.rr_tower = nn.Sequential(
            nn.Linear(flat_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # --- Prediction Heads ---
        # Each head takes the refined features from its tower and makes a prediction.

        self.main_head = nn.Linear(32, 2) # 2 classes for arrhythmia
        self.rr_head = nn.Linear(32, 1)   # 1 value for RR interval

        # --- Pretext Task Tower and Head (Conditional) ---
        if self.with_pretext_task:
            self.pretext_tower = nn.Sequential(
                nn.Linear(flat_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.pretext_head = nn.Linear(32, 6) # 6 classes for permutation

        # --- Learnable parameters for uncertainty-based loss weighting ---
        self.log_var_main = nn.Parameter(torch.tensor(0.0))
        self.log_var_rr = nn.Parameter(torch.tensor(0.0))
        if self.with_pretext_task:
            self.log_var_pretext = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # 1. Shared Feature Extraction
        shared_features = self.feature_extractor(x)
        flat_features = shared_features.view(shared_features.size(0), -1) # Flatten

        # 2. Pass features through task-specific towers
        main_refined = self.main_tower(flat_features)
        rr_refined = self.rr_tower(flat_features)

        # 3. Make predictions using the heads
        main_pred = self.main_head(main_refined)
        rr_pred = self.rr_head(rr_refined).squeeze(-1)

        # 4. Handle pretext task if active
        if self.with_pretext_task:
            pretext_refined = self.pretext_tower(flat_features)
            pretext_pred = self.pretext_head(pretext_refined)
            return (main_pred, rr_pred, pretext_pred,
                    self.log_var_main, self.log_var_rr, self.log_var_pretext)
        else:
            return main_pred, rr_pred, self.log_var_main, self.log_var_rr