

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an  ## uncomment for animation
plt.switch_backend("Qt5Agg")

# Define force of gravity

def F(m1, m2, pos1, pos2):
    # pos 1 and pos 2 are positions
    rr = (np.linalg.norm(pos2 - pos1))
    if(rr == 0) :
        return 0
    else :
        x = G * m1 * m2 * (pos2 - pos1)/(rr ** 3)  # input the correct function here
        return x


# ------------------------- PHYSICAL PARAMETERS ------------------------
n = 10
G = 6.7e-11  # Universal gravitational constant
m_sun = 1.989e30  # Mass of sun (kg)
m_earth = 5.972e24  # Mass of Earth (kg)
m_moon = 7.348e22  # Mass of Moon (kg)
m_mercury = 3.3011e23
m_venus = 4.8675e24
m_mars = 6.4171e23
m_jupiter = 1.89819e27
m_saturn = 5.6834e26
m_uranus = 8.6813e25
m_neptune = 1.02413e26
masses = np.array([m_sun,m_mercury,m_earth,m_moon,m_venus,m_mars,m_jupiter,m_saturn,m_uranus,m_neptune])
d_earth = 1.496e11  # Earth-Sun distance (m)
d_moon = 3.844e8  # Earth-Moon distance (m)
d_mercury = 5.791e10
d_venus = 1.0821e11
d_mars = 2.2792e11
d_jupiter = 7.7857e11
d_saturn = 1.43353e12
d_uranus = 2.87246e12
d_neptune = 4.49506e12
distance = np.array([0,d_mercury,d_earth,d_moon,d_venus,d_mars,d_jupiter,d_saturn,d_uranus,d_neptune])
v_earth_0 = 2.978e4  # Magnitude of velocity of Earth around the Sun (m/s)
v_moon_0 = 1.02e3  # Magnitude of velocity of Moon around the Earth (m/s)
v_mercury_0 = 4.736e4
v_venus_0 = 3.502e4
v_mars_0 = 2.407e4
v_jupiter_0 = 1.306e4
v_saturn_0 = 9.68e3
v_uranus_0 = 6.80e3
v_neptune_0 = 5.43e3
velocity =np.array([0,v_mercury_0,v_earth_0,v_moon_0,v_venus_0,v_mars_0,v_jupiter_0,v_saturn_0,v_uranus_0,v_neptune_0])
#dt = 100
dt = 1000  # Time step (s)
#t_end = 3.154e7
#t_end = 2e5
t_end = 7e7
n_timesteps = int(t_end / dt)
t = np.linspace(0., t_end, n_timesteps)
t_len = len(t)

r = np.zeros([n,t_len,2])
p = np.zeros([n,t_len,2])
# define initial values:

for i in range(1,n):

    r[i,0] = [distance[i],0.]
    p[i,0] = [0,velocity[i]*masses[i]]
r[3,0,0] += d_earth
p[3,0,1] += m_moon*v_earth_0

p[0,0] = [0,-1*np.cumsum(p[:,0,1])[n-1]]
print(p[0,0])
print(type(p[0,0,0]))
#for moon

#


for i in range(t_len - 1):
    for j in range(n):
        r[j,i+1] = r[j,i]+p[j,i]*dt/masses[j]
        p[j,i+1] = p[j,i]
        for k in range(n):
            p[j,i+1] += dt*F(masses[j],masses[k],r[j,i],r[k,i])

'''
print(r[0])
print(r[1])
print(r[2])
print(r[3])
print(p[0])
print(p[1])
print(p[2])
print(p[3])
plt.plot(r[0,:,0],r[0,:,1])
plt.plot(r[1,:,0],r[1,:,1])
plt.plot(r[2,:,0],r[2,:,1])
plt.plot(r[3,:,0],r[3,:,1])

for i in range(n):
    plt.plot(r[i,:,0],r[i,:,1])
plt.show()


plt.plot(r_earth[:,0],r_earth[:,1])
plt.plot(r_sun[:,0],r_sun[:,1])
plt.plot(r_moon[:,0],r_moon[:,1])
plt.show()
'''


def init():
    # print (lines[0])
    # a.set_data([],[])

    for i in range(n):
        lines[i][0].set_data([], [])
    temp = ()
    for i in range(n):
        temp += (lines[i][0],)
    return temp


# function to animate data; length of each line increases until the end is reached
def animate(i):
    for j in range(n):
        lines[j][0].set_data(coord[j, :i, 0], coord[j, :i, 1])
    temp = ()
    for j in range(n):
        temp += (lines[j][0],)
    return temp


# animated plot
# create figure
fig = plt.figure()
# define axes
ax = plt.axes(xlim=(-6e12, 6e12), ylim=(-6e12, 6e12))
# create lines (empty for now)
lines = ()
#lines = np.array(ax.plot([], [], lw=2) * (n))
#lines = np.array(lines)
for i in range(n):
    lines += (ax.plot([],[],lw=2),)
# print(lines[0][0][0])
# print(type(lines[0][0][0]))
# lines = np.array(ax.plot([],[],lw=2)*(n))

# lines = np.array(lines)


# a, = ax.plot([],[],lw=2)
# print(a)


# separate the data into x and y components for clarity
# also skip some indices to speed things up
skip = 100
coord = np.zeros([n, len(r[0, :, 0][::skip]), 2])

for i in range(n):
    coord[i, :, 0] = r[i, :, 0][::skip]

    coord[i, :, 1] = r[i, :, 1][::skip]

# number of frames in the animation
num_frames = len(coord[0, :, 0]) - 1
# print(num_frames)
# function to initialize lines for animation


# animate and plot
anim = an.FuncAnimation(fig, animate, init_func=init, blit=True, interval=25, frames=num_frames)
plt.show()

