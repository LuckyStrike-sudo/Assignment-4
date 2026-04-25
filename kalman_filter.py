import numpy as np
#This will contain all of the code for kalman filter class
class kalmanFilter:
    def __init__(self, initial_position, objId):
        self.x = np.array([initial_position[0], initial_position[1], 0, 0])  # [px, py, vx, vy]
        self.P = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # Initial covariance matrix
        self.D = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # state transition matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # measurement matrix
        self.Q = np.array([[0.1, 0, 0, 0],
                           [0, 0.1, 0, 0],
                           [0, 0, 0.1, 0],
                           [0, 0, 0, 0.1]])  # process noise covariance
        self.R = np.array([[0.1, 0],
                           [0, 0.1]])  # measurement noise covariance
        self.dt = 1  # time step
        self.objId = objId  # ID of the tracked object
        self.positions = []  # List to store the positions of the tracked object
        self.frames_since_update = 0  # Counter for frames since the last update
        self.active = True  # Flag to indicate if the tracker is active

    def predict(self):
        self.x = self.D @ self.x  # State prediction
        self.P = self.D @ self.P @ self.D.T + self.Q  # Covariance prediction

    def update(self, y):
        predicted = self.H @ self.x  # Predicted measurement
        error = y - predicted  # Measurement error
        uncertainty = self.H @ self.P @ self.H.T + self.R  # Measurement uncertainty
        gain = self.P @ self.H.T @ np.linalg.inv(uncertainty)  # Kalman gain
        self.x = self.x + gain @ error  # State update
        self.P = self.P - gain @ self.H @ self.P  # Covariance update
        self.positions.append((self.x[0], self.x[1]))  # Store the position
        self.frames_since_update = 0  # Reset the counter
