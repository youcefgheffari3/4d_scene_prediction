import torch
import torch.nn as nn

class TrajectoryGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1, dropout=0.1):
        super(TrajectoryGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, observed_seq, pred_len):
        """
        observed_seq: Tensor of shape (1, T_obs, 3)
        pred_len: int, number of future time steps to predict

        Returns:
            pred_seq: Tensor of shape (pred_len, 3)
        """
        output_seq = []
        h = None
        x = observed_seq  # shape (1, T_obs, 3)

        for _ in range(pred_len):
            out, h = self.gru(x, h)  # out: (1, T_obs, hidden)
            next_pos = self.fc(out[:, -1, :])  # (1, 3)
            output_seq.append(next_pos.squeeze(0))
            x = torch.cat([x[:, 1:], next_pos.unsqueeze(1)], dim=1)

        return torch.stack(output_seq, dim=1).squeeze(0)  # (pred_len, 3)
