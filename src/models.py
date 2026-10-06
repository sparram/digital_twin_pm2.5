import torch
import torch.nn as nn

class PINN_Aburra(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super(PINN_Aburra, self).__init__()
        
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 4)) # Salidas: [C, u, v, S]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y, t):
        # x, y, t deben ser tensores columna de PyTorch
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.net(inputs)
        
        C = outputs[:, 0:1] # Concentración de PM2.5
        u = outputs[:, 1:2] # Viento Zonal
        v = outputs[:, 2:3] # Viento Meridional
        S = outputs[:, 3:4] # Fuentes / Sumideros
        return C, u, v, S

print("-> Arquitectura PINN_Aburra cargada con éxito.")