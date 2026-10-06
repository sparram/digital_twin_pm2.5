import torch

def compute_pde_residuals(model, x, y, t, K=0.01):
    """Calcula los residuos de la ecuación de transporte y continuidad usando autograd."""
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    C, u, v, S = model(x, y, t)
    
    # Derivadas de C
    C_t = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_x = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    C_y = torch.autograd.grad(C, y, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    
    C_xx = torch.autograd.grad(C_x, x, grad_outputs=torch.ones_like(C_x), create_graph=True)[0]
    C_yy = torch.autograd.grad(C_y, y, grad_outputs=torch.ones_like(C_y), create_graph=True)[0]
    
    # Derivadas del viento (Continuidad)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    # 1. Residuo de Transporte (Advección-Difusión + Fuentes)
    res_transport = C_t + (u * C_x) + (v * C_y) - K * (C_xx + C_yy) - S
    
    # 2. Residuo de Continuidad (Fluido incompresible 2.5D)
    res_continuity = u_x + v_y
    
    return res_transport, res_continuity, S

def total_loss_fn(model, x_data, y_data, t_data, C_true, u_true, v_true, x_col, y_col, t_col, weights=(1.0, 0.1, 0.01)):
    """Calcula la función de pérdida total combinando datos supervisados y física."""
    w_data, w_pde, w_reg = weights
    
    # A. Pérdida Supervisada (Datos reales de estaciones)
    C_pred, u_pred, v_pred, _ = model(x_data, y_data, t_data)
    loss_data = torch.mean((C_pred - C_true) ** 2) + \
                torch.mean((u_pred - u_true) ** 2) + \
                torch.mean((v_pred - v_true) ** 2)
    
    # B. Pérdida Física (Residuos en puntos de colocación internos)
    res_trans, res_cont, S_pred = compute_pde_residuals(model, x_col, y_col, t_col)
    loss_pde = torch.mean(res_trans ** 2) + torch.mean(res_cont ** 2)
    
    # C. Regularización (Evitar emisiones negativas no físicas)
    loss_reg = torch.mean(torch.relu(-S_pred) ** 2)
    
    loss_total = (w_data * loss_data) + (w_pde * loss_pde) + (w_reg * loss_reg)
    return loss_total

print("-> Módulo de física y función de pérdida listo.")