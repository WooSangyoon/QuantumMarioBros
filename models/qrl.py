import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from pathlib import Path


class QRLModel(nn.Module):
    def __init__(self, input_dim=4, num_actions=0):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.n_wires = input_dim
        self.n_layers = 2
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.q_head = nn.Sequential(
            nn.Linear(self.n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

        self.rx_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ry_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(self.n_wires)]
                )
                for _ in range(self.n_layers)
            ]
        )

    def _run_quantum_layer(self, x):
        batch_size = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=x.device)

        for i in range(self.n_wires):
            tqf.ry(qdev, wires=i, params=x[:, i])

        for layer_idx in range(self.n_layers):
            for i in range(self.n_wires):
                tqf.cnot(qdev, wires=[i, (i + 1) % self.n_wires])

            for i in range(self.n_wires):
                self.rx_layers[layer_idx][i](qdev, wires=i)
                self.ry_layers[layer_idx][i](qdev, wires=i)

        measured = self.measure(qdev)
        return measured

    def export_circuit_diagram(self, path):
        import os

        project_root = Path(__file__).resolve().parent.parent
        mpl_config_dir = project_root / ".mplconfig"
        mpl_config_dir.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        qc = QuantumCircuit(self.n_wires, self.n_wires)
        input_angles = ParameterVector("x", self.n_wires)
        rx_params = ParameterVector("theta_rx", self.n_layers * self.n_wires)
        ry_params = ParameterVector("theta_ry", self.n_layers * self.n_wires)

        for wire in range(self.n_wires):
            qc.ry(input_angles[wire], wire)

        for layer_idx in range(self.n_layers):
            for wire in range(self.n_wires):
                qc.cx(wire, (wire + 1) % self.n_wires)

            for wire in range(self.n_wires):
                param_idx = layer_idx * self.n_wires + wire
                qc.rx(rx_params[param_idx], wire)
                qc.ry(ry_params[param_idx], wire)

            qc.barrier()

        qc.measure(range(self.n_wires), range(self.n_wires))
        figure = qc.draw(output="mpl")
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def forward(self, x):
        x = torch.tanh(x)
        x = self._run_quantum_layer(x)
        q_values = self.q_head(x)
        return q_values
