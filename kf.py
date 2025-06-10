import numpy as np
import matplotlib.pyplot as plt

dt = 1.0
num_steps = 50

u = np.array([0.1, -0.05])
x = np.zeros((4, num_steps))
x[:, 0] = [0.0, 1.0, 0.0, 1.5]

Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
R = np.diag([1.0, 1.0])

A = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])

B = np.array([[0.5*dt**2, 0],
              [dt,        0],
              [0, 0.5*dt**2],
              [0,        dt]])

H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

z = np.zeros((2, num_steps))

for k in range(1, num_steps):
    x[:, k] = A.dot(x[:, k-1]) + B.dot(u)
    x[:, k] += np.random.multivariate_normal(np.zeros(4), Q)
    z[:, k] = H.dot(x[:, k]) + np.random.multivariate_normal(np.zeros(2), R)

x_hat = np.zeros((4, num_steps))
P = np.eye(4)
x_hat[:, 0] = [0, 0, 0, 0]

cov_trace = []

for k in range(1, num_steps):
    x_pred = A.dot(x_hat[:, k-1]) + B.dot(u)
    P_pred = A.dot(P).dot(A.T) + Q

    K = P_pred.dot(H.T).dot(np.linalg.inv(H.dot(P_pred).dot(H.T) + R))
    x_hat[:, k] = x_pred + K.dot(z[:, k] - H.dot(x_pred))
    P = (np.eye(4) - K.dot(H)).dot(P_pred)

    cov_trace.append(np.trace(P))

plt.figure()
plt.plot(x[0], x[2], label='True', color='orange')
plt.plot(z[0], z[1], '.', label='Observations', color='blue', alpha=0.7)
plt.plot(x_hat[0], x_hat[2], '--', label='KF', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Newtonian Motion with KF')
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
