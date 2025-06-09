import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 參數設定
# -----------------------------
dt = 1.0
num_steps = 50

# 真實加速度 [ax, ay]
true_acceleration = np.array([0.1, -0.05])

# 狀態維度與量測維度
n = 4   # [x, xdot, y, ydot]
m = 2   # [x, y] 直接量測位置

# 系統噪聲與量測噪聲協方差
Q = np.diag([1e-4]*n)
R = np.eye(m) * 1.0**2

# 狀態轉移與控制
F = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])
G = np.array([[0.5*dt**2, 0],
              [dt,        0],
              [0, 0.5*dt**2],
              [0,        dt]])

# UKF 參數 (scaled unscented transform)
alpha = 1e-3
kappa = 0
beta  = 2.0
lambda_ = alpha**2 * (n + kappa) - n
gamma = np.sqrt(n + lambda_)

# sigma point 權重
Wm = np.full(2*n+1, 1.0/(2*(n+lambda_)))
Wc = np.full(2*n+1, 1.0/(2*(n+lambda_)))
Wm[0] = lambda_/(n+lambda_)
Wc[0] = lambda_/(n+lambda_) + (1 - alpha**2 + beta)

# -----------------------------
# 產生真實軌跡與量測
# -----------------------------
x_true = np.zeros((n, num_steps))
x_true[:,0] = [0.0, 1.0, 0.0, 1.5]
measurements = np.zeros((m, num_steps))

for k in range(1, num_steps):
    # 真實狀態
    x_true[:,k] = F @ x_true[:,k-1] + G @ true_acceleration \
                  + np.random.multivariate_normal(np.zeros(n), Q)
    # 量測位置 [x,y]
    measurements[:,k] = x_true[[0,2],k] + np.random.randn(m)

# -----------------------------
# UKF 初始化
# -----------------------------
x_hat = np.zeros((n, num_steps))
P = np.eye(n)
x_hat[:,0] = np.zeros(n)
cov_trace = []

# -----------------------------
# UKF 迴圈
# -----------------------------
for k in range(1, num_steps):
    # 1. 產生 sigma points
    Psqrt = np.linalg.cholesky(P)
    sigma = np.zeros((n, 2*n+1))
    sigma[:,0] = x_hat[:,k-1]
    for i in range(n):
        sigma[:, i+1   ] = x_hat[:,k-1] + gamma * Psqrt[:,i]
        sigma[:, i+1+n ] = x_hat[:,k-1] - gamma * Psqrt[:,i]

    # 2. 時間更新 (propagate through f)
    sigma_pred = np.zeros_like(sigma)
    for i in range(2*n+1):
        sigma_pred[:,i] = F @ sigma[:,i] + G @ true_acceleration
    x_pred = sigma_pred @ Wm         # 預測狀態
    P_pred = Q.copy()
    for i in range(2*n+1):
        diff = (sigma_pred[:,i] - x_pred).reshape(-1,1)
        P_pred += Wc[i] * (diff @ diff.T)

    # 3. 量測更新 (propagate through h)
    Z_sigma = sigma_pred[[0,2],:]    # h(x) = Hx, 直接取 x,y
    z_pred = Z_sigma @ Wm
    P_zz = R.copy()
    for i in range(2*n+1):
        dz = (Z_sigma[:,i] - z_pred).reshape(-1,1)
        P_zz += Wc[i] * (dz @ dz.T)

    P_xz = np.zeros((n,m))
    for i in range(2*n+1):
        dx = (sigma_pred[:,i] - x_pred).reshape(-1,1)
        dz = (Z_sigma[:,i] - z_pred).reshape(-1,1)
        P_xz += Wc[i] * (dx @ dz.T)

    # 卡爾曼增益
    K = P_xz @ np.linalg.inv(P_zz)

    # 更新
    z = measurements[:,k]
    x_hat[:,k] = x_pred + K @ (z - z_pred)
    P = P_pred - K @ P_zz @ K.T

    cov_trace.append(np.trace(P))

# -----------------------------
# 繪圖
# -----------------------------
plt.figure(figsize=(5,5))
plt.plot(x_true[0], x_true[2], '-', label='True')
plt.scatter(measurements[0], measurements[1], s=20, c='orange', label='Meas')
plt.plot(x_hat[0], x_hat[2], '--', label='UKF')
plt.xlabel('x'); plt.ylabel('y'); plt.legend()
plt.title('2D Motion with UKF'); plt.grid(); plt.axis('equal')

plt.figure(figsize=(5,3))
plt.plot(range(1,num_steps), cov_trace, '-o')
plt.xlabel('Step'); plt.ylabel('Trace(P)')
plt.title('UKF Uncertainty'); plt.grid()

plt.show()
