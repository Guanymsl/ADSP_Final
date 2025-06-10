import numpy as np
import matplotlib.pyplot as plt

dt = 1.0
num_steps = 50

u = np.array([0.1, -0.05])
x = np.zeros((4, num_steps))
x[:, 0] = [0.0, 1.0, 0.0, 1.5]

Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
R = np.diag([1.0, 0.05**2])

A = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])

B = np.array([[0.5*dt**2, 0],
              [dt,        0],
              [0, 0.5*dt**2],
              [0,        dt]])

n = 4
m = 2
alpha = 5 * 1e-3
kappa = 0
beta  = 2
lambda_ = alpha ** 2 * (n + kappa) - n
gamma = np.sqrt(n + lambda_)

Wm = np.full(2*n+1, 1.0/(2*(n+lambda_)))
Wc = np.full(2*n+1, 1.0/(2*(n+lambda_)))
Wm[0] = lambda_/(n+lambda_)
Wc[0] = lambda_/(n+lambda_) + (1 - alpha**2 + beta)

z = np.zeros((2, num_steps))

for k in range(1, num_steps):
    x[:, k] = A.dot(x[:, k-1]) + B.dot(u)
    x[:, k] += np.random.multivariate_normal(np.zeros(4), Q)

    px, vx, py, vy = x[:, k]
    z[0, k] = np.hypot(px, py)
    z[1, k] = np.arctan2(py, px)
    z[:, k] += np.random.multivariate_normal(np.zeros(2), R)

x_hat = np.zeros((4, num_steps))
P = np.eye(4)
x_hat[:, 0] = [0, 0, 0, 0]

cov_trace = []

for k in range(1, num_steps):
    Psqrt = np.linalg.cholesky(P)
    sigma = np.zeros((n, 2*n+1))
    sigma[:,0] = x_hat[:,k-1]
    for i in range(n):
        sigma[:, i+1   ] = x_hat[:,k-1] + gamma * Psqrt[:,i]
        sigma[:, i+1+n ] = x_hat[:,k-1] - gamma * Psqrt[:,i]

    sigma_pred = np.zeros_like(sigma)
    for i in range(2*n+1):
        sigma_pred[:,i] = A.dot(sigma[:,i]) + B.dot(u)
    x_pred = sigma_pred.dot(Wm)
    P_pred = Q.copy()
    for i in range(2*n+1):
        diff = (sigma_pred[:,i] - x_pred).reshape(-1,1)
        P_pred += Wc[i] * (diff.dot(diff.T))

    Z_sigma = np.vstack([np.sqrt(sigma_pred[0, :]**2 + sigma_pred[2, :]**2),
                         np.arctan2(sigma_pred[2, :], sigma_pred[0, :])])
    z_pred = Z_sigma.dot(Wm)
    P_zz = R.copy()
    for i in range(2*n+1):
        dz = (Z_sigma[:,i] - z_pred).reshape(-1,1)
        P_zz += Wc[i] * (dz @ dz.T)

    P_xz = np.zeros((n,m))
    for i in range(2*n+1):
        dx = (sigma_pred[:,i] - x_pred).reshape(-1,1)
        dz = (Z_sigma[:,i] - z_pred).reshape(-1,1)
        P_xz += Wc[i] * (dx @ dz.T)

    K = P_xz.dot(np.linalg.inv(P_zz))
    x_hat[:,k] = x_pred + K.dot((z[:,k] - z_pred))
    P = P_pred - K.dot(P_zz).dot(K.T)

    cov_trace.append(np.trace(P))

plt.figure()
plt.plot(x[0], x[2], label='True', color='orange')
meas_x = z[0] * np.cos(z[1])
meas_y = z[0] * np.sin(z[1])
plt.scatter(meas_x, meas_y, s=10, label='Observations', color='blue', alpha=0.7)
plt.plot(x_hat[0], x_hat[2], '--', label='UKF', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Newtonian Motion with UKF')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.figure()
plt.plot(range(1, num_steps), cov_trace, color='purple')
plt.xlabel('Time Step')
plt.ylabel('Trace of Covariance Matrix')
plt.title('Error')
plt.grid(True)
plt.show()
