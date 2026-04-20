import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, input_shape=(4, 84, 84), num_actions=0):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(self.input_shape)

        self.q_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        out = self.features(x)
        conv_out_size = int(torch.flatten(out, start_dim=1).shape[1])
        return conv_out_size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        q_values = self.q_head(x)
        return q_values
