import torch
import torch.optim as optim
from tqdm.auto import tqdm
from src.models import PINN_Aburra
from src.physics import total_loss_fn

def train_pinn(x_data, y_data, t_data, C_true, u_true, v_true, 
               x_col, y_col, t_col, epochs=2000, lr=0.001):
    
    # 1. Inicializar el modelo
    model = PINN_Aburra(hidden_dim=128, num_layers=3)
    
    # 2. Configurar el optimizador
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("-> Iniciando entrenamiento del PINN...")
    
    # Envolvemos el rango con tqdm para ver la barra de progreso
    for epoch in tqdm(range(epochs), desc="Entrenando PINN"):
        optimizer.zero_grad()
        
        # Calcular la pérdida total usando la función física
        loss = total_loss_fn(
            model, x_data, y_data, t_data, 
            C_true, u_true, v_true, 
            x_col, y_col, t_col, 
            weights=(1.0, 0.1, 0.01)
        )
        
        # Retropropagación
        loss.backward()
        
        # Actualizar los pesos
        optimizer.step()
        
        # Usamos tqdm.write() en lugar de print() para que no rompa la barra de progreso
        if (epoch + 1) % 200 == 0:
            tqdm.write(f"Época [{epoch+1}/{epochs}] | Pérdida Total: {loss.item():.6f}")
            
    print("-> ¡Entrenamiento finalizado con éxito!")
    return model