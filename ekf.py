import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
N  = 10000

u_true = np.zeros((3, N))
for k in range(N):
    u_true[:, k] = [0, 0.1, np.deg2rad(0.1)]

sigma_v     = 0.1
sigma_omega = np.deg2rad(0.01)
Q = np.diag([sigma_v**2, sigma_v**2, sigma_omega**2])

sigma_r = 0.5
R = np.array([[sigma_r**2]])

x_ideal = np.zeros((3, N))
x_true  = np.zeros((3, N))
x_true[:, 0] = [0.0, 0.0, 0.0]

z_meas = np.zeros(N)
z_meas[0] = 0.0

x_est = np.zeros((3, N))
P_est = np.zeros((3, 3, N))

x_est[:, 0] = [0.0, 0.0, 0.0]
P_est[:, :, 0] = np.eye(3) * 0.1

error_history = np.zeros(N)

for k in range(1, N):
    v_x   = u_true[0, k-1]
    v_y   = u_true[1, k-1]
    omega = u_true[2, k-1]
    theta_ideal_prev = x_ideal[2, k-1]
    theta_true_prev  = x_true[2, k-1]

    vx_world_ideal = v_x * np.sin(theta_ideal_prev) + v_y * np.cos(theta_ideal_prev)
    vy_world_ideal = -v_x * np.cos(theta_ideal_prev) + v_y * np.sin(theta_ideal_prev)
    x_ideal[0, k] = x_ideal[0, k-1] + vx_world_ideal * dt
    x_ideal[1, k] = x_ideal[1, k-1] + vy_world_ideal * dt
    x_ideal[2, k] = x_ideal[2, k-1] + omega * dt

    vx_world = v_x * np.sin(theta_true_prev) + v_y * np.cos(theta_true_prev)
    vy_world = -v_x * np.cos(theta_true_prev) + v_y * np.sin(theta_true_prev)
    x_true[0, k] = x_true[0, k-1] + vx_world * dt + sigma_v * np.random.randn()
    x_true[1, k] = x_true[1, k-1] + vy_world * dt + sigma_v * np.random.randn()
    x_true[2, k] = x_true[2, k-1] + omega * dt + sigma_omega * np.random.randn()

    true_range = np.sqrt(x_true[0, k] ** 2 + x_true[1, k] ** 2)
    z_meas[k] = true_range + sigma_r * np.random.randn()

    x_prev = x_est[:, k-1].copy()
    P_prev = P_est[:, :, k-1].copy()

    v_x_est   = u_true[0, k-1]
    v_y_est   = u_true[1, k-1]
    omega_est = u_true[2, k-1]
    theta_e_prev = x_prev[2]

    vx_world_e = v_x_est * np.sin(theta_e_prev) + v_y_est * np.cos(theta_e_prev)
    vy_world_e = -v_x_est * np.cos(theta_e_prev) + v_y_est * np.sin(theta_e_prev)

    x_pred = np.zeros(3)
    x_pred[0] = x_prev[0] + vx_world_e * dt
    x_pred[1] = x_prev[1] + vy_world_e * dt
    x_pred[2] = x_prev[2] + omega_est * dt

    F = np.eye(3)
    F[0, 2] = (v_x_est * np.cos(theta_e_prev) - v_y_est * np.sin(theta_e_prev)) * dt
    F[1, 2] = (v_x_est * np.sin(theta_e_prev) + v_y_est * np.cos(theta_e_prev)) * dt

    P_pred = F @ P_prev @ F.T + Q

    hx = np.sqrt(x_pred[0]**2 + x_pred[1]**2)

    if hx < 1e-6:
        H = np.array([[0.0, 0.0, 0.0]])
    else:
        H = np.array([[x_pred[0]/hx, x_pred[1]/hx, 0.0]])

    S = H @ P_pred @ H.T + R
    K = (P_pred @ H.T) / S

    zk = z_meas[k]

    x_upd = x_pred + (K.flatten() * (zk - hx))
    P_upd = (np.eye(3) - K @ H) @ P_pred

    x_est[:, k]   = x_upd
    P_est[:, :, k] = P_upd

    error_history[k] = np.linalg.norm(x_true[0:2, k] - x_upd[0:2])

plt.figure(figsize=(6, 6))
plt.plot(x_ideal[0, :], x_ideal[1, :], 'g-', linewidth=2, label='Ideal (noiseless)')
plt.plot(x_true[0, :],  x_true[1, :],  'b-', linewidth=1, label='Noisy Truth')
plt.plot(x_est[0, :],   x_est[1, :],   'r--', linewidth=2, label='EKF Estimate')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectories: Ideal vs. Noisy Truth vs. EKF')
plt.legend(loc='best')
plt.grid(True)
plt.axis('equal')

time = np.arange(N) * dt
plt.figure(figsize=(6, 4))
plt.plot(time, error_history, 'k-', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Position Error vs. Time')
plt.grid(True)

plt.show()
