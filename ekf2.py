import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 參數設定
# -----------------------------
dt = 1.0
num_steps = 50

# 真實加速度 [ax, ay]
true_acceleration = np.array([0.1, -0.05])

# 狀態向量 [x, xdot, y, ydot]
x_true = np.zeros((4, num_steps))
x_true[:, 0] = [0.0, 1.0, 0.0, 1.5]

# 系統雜訊協方差 Q
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])

# 量測雜訊標準差
sigma_r = 1.0       # 範圍雜訊
sigma_theta = 0.05  # 方位雜訊 (rad)
R = np.diag([sigma_r**2, sigma_theta**2])

# 狀態轉移與控制矩陣
F = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])
G = np.array([[0.5*dt**2, 0],
              [dt,        0],
              [0, 0.5*dt**2],
              [0,        dt]])

# -----------------------------
# 產生真實軌跡並模擬量測
# -----------------------------
measurements = np.zeros((2, num_steps))  # [range; bearing]

for k in range(1, num_steps):
    # 真實狀態更新
    x_true[:, k] = F @ x_true[:, k-1] \
                   + G @ true_acceleration \
                   + np.random.multivariate_normal(np.zeros(4), Q)
    # 量測：範圍 r、方位 theta
    px, vx, py, vy = x_true[:, k]
    r_true = np.hypot(px, py)
    theta_true = np.arctan2(py, px)
    measurements[0, k] = r_true + np.random.randn() * sigma_r
    measurements[1, k] = theta_true + np.random.randn() * sigma_theta

# -----------------------------
# EKF 初始化
# -----------------------------
x_hat = np.zeros((4, num_steps))
P = np.eye(4) * 1.0
x_hat[:, 0] = [0.1, 0.0, 0.1, 0.0]  # 初始狀態
cov_trace = []

# -----------------------------
# EKF 迴圈
# -----------------------------
for k in range(1, num_steps):
    # 1. Prediction
    x_pred = F @ x_hat[:, k-1] + G @ true_acceleration
    P_pred = F @ P @ F.T + Q

    # 2. Measurement prediction
    px, vx, py, vy = x_pred
    r_pred = np.hypot(px, py)
    theta_pred = np.arctan2(py, px)
    h = np.array([r_pred, theta_pred])

    # 3. 計算雅可比 H_j
    #    避免 r_pred 為 0
    if r_pred < 1e-6:
        r_pred = 1e-6
    H_j = np.array([
        [ px/r_pred,      0, py/r_pred,       0 ],
        [ -py/r_pred**2,  0, px/r_pred**2,    0 ]
    ])  # shape (2,4)

    # 4. Innovation
    z = measurements[:, k]
    y_res = z - h
    # 對角度差做 wrap
    y_res[1] = (y_res[1] + np.pi) % (2*np.pi) - np.pi

    # 5. 卡爾曼增益
    S = H_j @ P_pred @ H_j.T + R
    K = P_pred @ H_j.T @ np.linalg.inv(S)

    # 6. 更新狀態與協方差
    x_hat[:, k] = x_pred + K @ y_res
    P = (np.eye(4) - K @ H_j) @ P_pred

    # 7. 紀錄 trace(P)
    cov_trace.append(np.trace(P))

# -----------------------------
# 畫圖
# -----------------------------
# 1) 真實軌跡 vs. 量測 (投影點) vs. EKF 估計
plt.figure(figsize=(6,6))
plt.plot(x_true[0], x_true[2],  '-', label='True Trajectory')
# 將 range, bearing 投影回 Cartesian 平面以便顯示
meas_x = measurements[0] * np.cos(measurements[1])
meas_y = measurements[0] * np.sin(measurements[1])
plt.scatter(meas_x, meas_y, s=20, c='orange', label='Measurements')
plt.plot(x_hat[0], x_hat[2], '--', label='EKF Estimate')
plt.xlabel('x'); plt.ylabel('y')
plt.title('2D Motion with Range＋Bearing EKF')
plt.legend(); plt.grid(True); plt.axis('equal')

# 2) 協方差 trace 隨時間變化
plt.figure(figsize=(6,3))
plt.plot(range(1,num_steps), cov_trace, '-o')
plt.xlabel('Time Step'); plt.ylabel('Trace(P)')
plt.title('Uncertainty Over Time')
plt.grid(True)

plt.show()
