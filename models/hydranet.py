import torch
import torch.nn as nn

class HydraNet(nn.Module):
    def __init__(self, feature_extractor_instance, with_pretext_task=False, with_rr_task=True):
        super().__init__()
        self.feature_extractor = feature_extractor_instance
        self.with_pretext_task = with_pretext_task
        self.with_rr_task = with_rr_task

        # --- Shared Feature Dimension ---
        flat_feature_dim = self.feature_extractor.output_channels * self.feature_extractor.output_length

        # --- Task-Specific Towers ---
        self.main_tower = nn.Sequential(
            nn.Linear(flat_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.main_head = nn.Linear(32, 2)

        if self.with_rr_task:
            self.rr_tower = nn.Sequential(
                nn.Linear(flat_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.rr_head = nn.Linear(32, 1)

        if self.with_pretext_task:
            self.pretext_tower = nn.Sequential(
                nn.Linear(flat_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.pretext_head = nn.Linear(32, 6)

        # --- Learnable parameters for uncertainty-based loss weighting ---
        self.log_var_main = nn.Parameter(torch.tensor(0.0))
        if self.with_rr_task:
            self.log_var_rr = nn.Parameter(torch.tensor(0.0))
        if self.with_pretext_task:
            self.log_var_pretext = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        shared_features = self.feature_extractor(x)
        flat_features = shared_features.view(shared_features.size(0), -1)

        main_refined = self.main_tower(flat_features)
        main_pred = self.main_head(main_refined)

        outputs = [main_pred]
        log_vars = [self.log_var_main]

        if self.with_rr_task:
            rr_refined = self.rr_tower(flat_features)
            rr_pred = self.rr_head(rr_refined).squeeze(-1)
            outputs.append(rr_pred)
            log_vars.append(self.log_var_rr)

        if self.with_pretext_task:
            pretext_refined = self.pretext_tower(flat_features)
            pretext_pred = self.pretext_head(pretext_refined)
            outputs.append(pretext_pred)
            log_vars.append(self.log_var_pretext)

        return tuple(outputs + log_vars)