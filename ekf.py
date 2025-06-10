import numpy as np
import matplotlib.pyplot as plt

dt = 1.0
num_steps = 50

u = np.array([0.1, -0.05])
x = np.zeros((4, num_steps))
x[:, 0] = [0.0, 1.0, 0.0, 1.5]

Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
R = np.diag([1.0, 0.05 ** 2])

A = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])

B = np.array([[0.5*dt**2, 0],
              [dt,        0],
              [0, 0.5*dt**2],
              [0,        dt]])

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
    x_pred = A.dot(x_hat[:, k-1]) + B.dot(u)
    P_pred = A.dot(P).dot(A.T) + Q

    px, vx, py, vy = x_pred
    r_pred = np.hypot(px, py)
    th_pred = np.arctan2(py, px)
    h_pred = np.array([r_pred, th_pred])

    if r_pred < 1e-6:
        r_pred = 1e-6

    H_j = np.array([[ px/r_pred,    0, py/r_pred,    0],
                    [-py/r_pred**2, 0, px/r_pred**2, 0]])

    y_res = z[:, k] - h_pred
    y_res[1] = (y_res[1] + np.pi) % (2 * np.pi) - np.pi

    K = P_pred.dot(H_j.T).dot(np.linalg.inv(H_j.dot(P_pred).dot(H_j.T) + R))
    x_hat[:, k] = x_pred + K.dot(y_res)
    P = (np.eye(4) - K.dot(H_j)).dot(P_pred)

    cov_trace.append(np.trace(P))

plt.figure()
plt.plot(x[0], x[2], label='True', color='orange')
meas_x = z[0] * np.cos(z[1])
meas_y = z[0] * np.sin(z[1])
plt.scatter(meas_x, meas_y, s=10, label='Observations', color='blue', alpha=0.7)
plt.plot(x_hat[0], x_hat[2], '--', label='EKF', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Newtonian Motion with EKF')
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
