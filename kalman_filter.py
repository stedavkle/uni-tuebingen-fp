import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(0, 2 * np.pi, 100)  # Sin wave periods
    measurements = np.concatenate([
        np.sin(x),  # 1st period
        np.sin(x),  # 2nd period
        np.sin(x),  # 3rd period
        np.log(np.linspace(1, np.exp(1), len(x) // 4))  # Logarithmic part
    ]) + np.random.normal(0, 0.1, len(x) * 3 + len(x) // 4)  # Adding noise

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []

    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()

def example2():
    # Time setup
    total_steps = 2000
    bits_per_finestep = 4

    dt = 1.0 / 60
    t_sine = np.linspace(0, 4*np.pi, int(total_steps/3*2))  # 2 periods of sine wave
    t_log = np.linspace(1, 100, int(total_steps/3))  # Logarithmic convergence time
    
    # Control input setup
    control_signal = 80 * np.sin(t_sine)  # Sine wave for the first part
    # Adjusted logarithmic convergence
    # Calculate starting point for the logarithmic part to ensure smooth transition
    log_start = control_signal[-1]
    # Generate logarithmic convergence towards -20
    control_convergence = log_start + (np.log(t_log) / np.log(t_log[-1])) * (-20 - log_start)
    control_input = np.concatenate((control_signal, control_convergence))
    
    
    # System dynamics matrices and Kalman Filter setup (same as before)
    F = np.array([[1, dt], [0, 1]])
    B = np.array([0.5*dt**2, dt]).reshape(-1, 1)
    H = np.array([1, 0]).reshape(1, 2)
    Q = np.array([[0.05, 0.05], [0.05, 0.05]])
    R = np.array([5]).reshape(1, 1)
    x0 = np.array([0, 0]).reshape(-1, 1)
    
    measurements = control_input
    min_val, max_val = -80, 80
    levels = np.linspace(min_val, max_val, bits_per_finestep, endpoint=False)[1:]
    measurements_quantized = np.digitize(measurements, levels)

    flip_probability = 0.5
    flip_distance = 100
    print(len(control_input))
    for i, meas in enumerate(measurements):
        distance_to_nearest_level = np.abs(meas - (min_val + (max_val - min_val) / 4 * (measurements[i])))
        if distance_to_nearest_level < flip_distance and np.random.rand() < flip_probability:
            direction = np.sign(meas - (min_val + (max_val - min_val) / 4 * (measurements[i])))
            measurements_quantized[i] = (measurements_quantized[i] + direction) % 4
    measurements = min_val + (max_val - min_val) / 4 * (measurements_quantized + 0.5) + np.random.normal(0, 0.8, len(control_input))

    # Kalman Filter processing
    kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, x0=x0)
    predictions = []
    for i in range(len(control_input)):
        u = np.array([control_input[i]])
        kf.predict(u=u)
        kf.update(measurements[i])
        predictions.append(np.dot(H, kf.x)[0])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(control_input, label='Control Input with Logarithmic Convergence')
    plt.plot(range(len(measurements)), measurements, color='red', label='2-bit Noisy Measurements with Bit Flips', alpha=0.5)
    plt.plot(predictions, label='Kalman Filter Prediction', linewidth=2)
    plt.ylim(-85, 85)
    plt.legend()
    plt.title("Kalman Filter with Logarithmic Convergence")
    plt.show()


if __name__ == '__main__':
    example2()