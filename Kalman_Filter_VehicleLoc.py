# this code isused for finding the 
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import glob

# location of a vehicle based on some sensor measurments
# State Extrapolation Equation xn1n = F @ xnn + G @ unn
# initial Conditions
dt = 1  # The time step between each measurement is 1 sec
siga = 0.2 # Uncertainity in random acceleration values
sigx = 3   # The measurement standard deviation in x
sigy = 3   # The measurement standard deviation in y
# The truth value is obtained from the car driviing in the direction
x1 = np.linspace(-400,0) 
x2 = np.linspace(0,300) 
#x = np.linspace(0,300)
y1 =  np.ones(len(x1))*300
y2 = np.sqrt((300)**2 - np.multiply(x2,x2))
x = np.append(x1,x2)
y = np.append(y1,y2)

# measurement values 
z = np.array([[-393.66, 300.4],[-375.93, 301.78],[-351.04, 295.1],[-328.96, 305.19],[-299.35, 301.06],[-273.36, 302.05],[-245.89, 300],[-222.58, 303.57],[-198.03, 296.33],[-174.17, 297.65],[-146.32, 297.41],[-123.72, 299.61],[-103.47, 299.6],[-78.23, 302.39],[-52.63, 295.04],[-23.34, 300.09],[25.96, 294.72],[49.72, 298.61],[76.94, 294.64],[95.38, 284.88],[119.83, 272.82],[144.01, 264.93],[161.84, 251.46],[180.56, 241.27],[201.42, 222.98],[222.62, 203.73],[239.4, 184.1],[252.51, 166.12],[266.26, 138.71],[271.75, 119.71],[277.4, 100.41],[294.12, 79.76],[301.23, 50.62],[291.8, 32.99],[299.89, 2.14],])
zx, zy = z.T

# print(z.T)
# path = r"C:\Users\in0119\Documents\VS_Code Positioning IN0119\Random_Codes"
# csv_files = glob.glob(os.path.join(path, "*.txt"))
# # print(csv_files)
# #Reading through  the files in the folder 
# for f in csv_files:
#     # read the csv file
#     df = pd.read_csv(f)
#     df.columns = ['X','Y']
    
# zx = np.array(df.X)           # x location measurement 
# zy = np.array(df.Y)           # Y location Measurement
xnn  = np.array([[0],
        [0],
        [0],
        [0],
        [0],
        [0]])    # Initial Conditions

F = np.array([[1,dt,0.5*dt**2,0,0,0],
    [0,1,dt,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,1,dt,0.5*dt**2],
    [0,0,0,0,1,dt],
    [0,0,0,0,0,1]])

# xnn = F @ xnn     # State Extrapolation Equation 
# Covariance etrapolation equation 
Q = np.array([[0.25*dt**4,0.5*dt**3,0.5*dt**2,0,0,0],
    [0.5*dt**3,dt**2,dt,0,0,0],
    [0.5*dt**2,dt,1,0,0,0],
    [0,0,0,0.25*dt**4,0.5*dt**3,0.5*dt**2],
    [0,0,0,0.5*dt**3,dt**2,dt],
    [0,0,0,0.5*dt**2,dt,1]])
Q = Q*siga
P = np.diag([500,500,500,500,500,500])

# Identity Matrix for the filter
I = np.identity(6)

#avar = 0.001    # random variance in acceleration 
# Pnn = F @ Pnn @ F.T + Q

H = np.array([[1,0,0,0,0,0],
    [0,0,0,1,0,0]])           # Observation matrix 

# Measurement Equation
#znn = H @ xnn + vn
# measurement Uncertainity Rn
R = np.array([[9,0],[0,9]])
xe = []
ye = []
vxe = []
vye = []
axe = []
aye = []
gp = []
gv = []
ga = []
for i in range(len(zx)):
    xnn = F @ xnn
    P = F @ P @ F.T + Q
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)               # Kalman Gain
    xnn = xnn + K @ (z[i] - H @ xnn)                           # State Update Equation
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T          # Covariance update Equation
    xest, yest = xnn.T
    kx, ky = K.T
    xe.append(xest[0])
    ye.append(yest[0])
    vxe.append(xest[1])
    vye.append(yest[1])
    axe.append(xest[2])
    aye.append(yest[2])
    gp.append(kx[0])
    gv.append(kx[1])
    ga.append(kx[2])

# Plotting Trugth measurement and estimation in the plot
fig, ax = plt.subplots(2)
ax[0].plot(x,y,label = 'Truth',color = 'green')
ax[0].plot(xe,ye,label = 'Estimate',color='red')
ax[0].scatter(xe,ye,color = 'red')
ax[0].plot(zx,zy,label = 'Measurement',color = 'blue')
ax[0].scatter(zx,zy,color = 'blue')
ax[0].legend(loc = 'lower left')
# ax[1].plot(vxe,vye,label = 'Velocity Estimate',color = 'green')
# ax[2].plot(axe,aye,label = 'Acceleration Estimate',color = 'blue')
# ax[1].legend(loc = 'upper right')
ax[1].plot(gp,label = 'Position Kalman Gain(Kp)',color = 'red')
# ax[1].plot(gv,label = 'Velocity Kalman Gain(Kv)',color = 'green')
# ax[1].plot(ga,label = 'Acceleration Kalman Gain(Ka)',color = 'blue')
ax[1].legend(loc = 'upper right')
plt.show()    




