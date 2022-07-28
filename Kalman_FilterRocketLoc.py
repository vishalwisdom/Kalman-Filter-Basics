# this code is used for estimating the location of the rocket 
import numpy as np
import matplotlib.pyplot as plt

# initial values
dt = 0.25 # time step between the mesurements in sec
ann = 30  # Rocket acceleration measurment in m/sec2
sigx = 20 # altimeter measurement error standard deviation m
eta = 0.1 # acceleraometer measurement error standard deviation m/sec2

# Measurements from altimeter of the rocket
z = np.array([-32.4,-11.1,18.01,22.9,19.5,28.5,46.5,68.9,48.2,56.1,90.5,104.9,140.9,148.01,187.6,209.2,244.6,276.4,323.5,357.3,357.4,398.3,446.7,465.1,529.4,570.4,636.8,693.3,707.3,748.5])

# The inital cotrol variable 
uo = 9.8   # acceleration due to gravity m/sec2
# Control inputs from the accelerometer of the rocket
u = np.array([39.72,40.02,39.97,39.81,39.75,39.6,39.77,39.83,39.73,39.87,39.81,39.92,39.78,39.98,39.76,39.86,39.61,39.86,39.74,39.87,39.63,39.67,39.96,39.8,39.89,39.85,39.9,39.81,39.81,39.68])
# initializing the true values of the rocket
t = np.linspace(0,8,30)
y = 0.5*30*(t**2)

# Iitialisation
x = np.array([[0],
              [0]])
x.shape = (2,1)

# Estimate uncertainity in rocket altitude 
P = np.diag([500,500])

# Observation matrix 
H = np.array([1,0])
H.shape = (1,2)

# identity Matrix 
I = np.identity(2)

F = np.array([[1,dt],
              [0,1]])          # State Transition matrix for Rocket

G = np.array([[0.5*dt**2],
              [dt]])           # Control matrix of the Rocket

Q = np.array([[0.25*dt**4,0.5*dt**3],
              [0.5*dt**3,dt**2]])        # process noise matric of the rocket
Q = Q * eta

R = sigx**2                              #Measurement Uncertainity matrix

# Initialising the system 
x = F @ x + G*uo
P = F @ P @ F.T + Q

# creating list to store all the values in the system
alte = []
ve = []
galte = []
gve = []

for i in range(len(z)):
    K = P @ (H.T * (H @ P @ H.T + R)**-1)                            # Kalman Gain
    x = x + K * (z[i] - H.dot(x))                                    # State Update Equation
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ (R * K.T)              # Covariance update Equation
    x = F @ x + G*u[i]
    P = F @ P @ F.T + Q
    xest, yest = x
    kx, ky = K
    alte.append(xest)
    ve.append(yest)
    galte.append(kx)
    gve.append(ky)

# plotting original and filtered data

fig, ax = plt.subplots(2)
ax[0].plot(t,y,label = 'Truth',color = 'green')
ax[0].plot(t,alte,label = 'Estimate',color='red')
ax[0].scatter(t,alte,color = 'red')
ax[0].plot(t,z,label = 'Measurement',color = 'blue')
ax[0].scatter(t,z,color = 'blue')
ax[0].legend(loc = 'upper left')
ax[1].set_xlabel('Time (sec)')
ax[0].set_ylabel('Altitude (m)')
ax[0].set_title('Rocket Altitude')
ax[1].set_ylabel('Kalman Gain')
ax[1].plot(t,galte,label = 'Position Kalman Gain (Kp)',color = 'black')
ax[1].legend(loc = 'upper right')
ax[1].grid(which = 'minor')
plt.show()
    
