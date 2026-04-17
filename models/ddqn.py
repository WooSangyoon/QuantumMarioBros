"""Double DQN baseline용 Q-network를 정의하는 파일."""

import torch
import torch.nn as nn
from actions import SIMPLE_MOVEMENT

class DDQN(nn.Module):
    def __init__(self, input_shape=(4, 84, 84)):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = len(SIMPLE_MOVEMENT)

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
        # TODO: x shape가 [B, C, H, W]인지 확인합니다.
        # env에서 batch 없는 state 하나를 넣는다면, agent 쪽에서 batch 차원을 추가할 수도 있습니다.
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        q_values = self.q_head(x)
        return q_values

# TODO: 구현할 때 확인할 것
# 1. 입력 state의 shape가 실제 env 출력과 맞는지
# 2. 채널 위치가 [C, H, W]인지
# 3. batch가 들어오면 [B, C, H, W]로 처리되는지
# 4. 출력 action 수가 SIMPLE_MOVEMENT 개수와 맞는지


# TODO: 다음 단계에서 이 파일은 agent가 사용하게 됩니다.
# 예: agent.select_action(state), agent.update()
