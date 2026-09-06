# Digital Twin: Evolution of Air Quality PM2.5 on Medellín Metropolitan Area

**Author:** Santiago Parra  
**Year:** 2026

<p align="center">
  <img src="media/aburra_pm25_enkf.gif" width="700" alt="PM2.5 ENKF Simulation">
</p>

This project uses an advection-diffusion model and point sensor data of air quality of Medellín to reconstruct the field of fine particulate matter (PM2.5) on the Aburrá Valley via an Ensemble Kalman Filter.

## Formulation of the Problem

Given a set of sensor measurements in time we want to reconstruct a field of PM2.5 concentration in the air of the city.
For this purpose, we use the SIATA measurements from June 2026 provided in [1] and we use a Data Assimilation Approach. We assume the dynamics of the PM2.5 concentration of fine particulate matter $c(x,t)$ is ruled according to the stochastic advection-diffusion model:

$$\frac{\partial c}{\partial t} + \mathbf{v} \cdot \nabla c = D \nabla^2 c + \nu_t$$

Where $\mathbf{v}$ represents the vector field that rules the wind velocity (transporting the clouds of PM2.5) and $D$ is the diffussion coeficient of the PM2.5 in the air. For the boundary conditions we impose *Neumann's Boundary Conditions* in the entire boundary to allow the particles to get in and get out of the domain easily. Finally, we use an initial condition $c(x,0)=0$ which is chosen by design and doesn't affect the fidelity of the experiment (as the diffusion term will neglect it in a small time period).

We would usually add the term $S(\mathbf{x}, t)$ at the rhs of the equation, which represents the source that generates the PM2.5 into the air (factories, cars moving around the city, etc). However, as it's unknown in the practice, we will compensate it by assimilating the sensor measurements. As time passes, we run the model and assimilate the sensor measurements using a Ensemble Kalman Filter, in which we correct the actual dynamics of the model according to the Kalman innovation equation:

$$c_{analysis} = c_{prior} + \mathbf{K}(y_{meas} - H c)$$

Where $H$ is the measurement operator (which takes the values of the sensor locations) and $\mathbf{K}$ is the Kalman gain, which is computed / approximated using the Ensemble simulations. That's why we also introduce the term $\nu_t$, which is a spatially-correlated noise that introduces variance to the model and avoids the ensemble's variance to collapse.

## Project Structure

```text
├── media/
│   └── aburra_pm25_enkf.gif     # Simulation visualization
├── src/
│   ├── physics_jax.py           # Advection-diffusion PDE solver & wind fields
│   └── enkf_jax.py              # EnKF implementation & Gaspari-Cohn localization
├── main.py                      # Main assimilation loop & GIF generation
├── requirements.txt             # Project dependencies
└── README.md
```

## References
[1] SIATA, 2026, "Material Particulado - PM2.5", https://doi.org/10.83041/AUWZWT, Repositorio de Datos SIATA, V4, UNF:6:5LlbSLDr/jhS2zLCY2F/Tw== [fileUNF]
