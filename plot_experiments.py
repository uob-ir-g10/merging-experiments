import matplotlib.pyplot as plt
import numpy as np

split_no_eps = np.loadtxt('split_no_eps.csv', delimiter=',')
split_theta_2 = np.loadtxt('split_theta_2.csv', delimiter=',')
split_theta_3 = np.loadtxt('split_theta_3.csv', delimiter=',')

all_no_eps = np.loadtxt('all_no_eps.csv', delimiter=',')
all_theta_2 = np.loadtxt('all_theta_2.csv', delimiter=',')
all_theta_3 = np.loadtxt('all_theta_3.csv', delimiter=',')

almost_no_eps = np.loadtxt('almost_no_eps.csv', delimiter=',')
almost_theta_2 = np.loadtxt('almost_theta_2.csv', delimiter=',')
almost_theta_3 = np.loadtxt('almost_theta_3.csv', delimiter=',')


fig, ax = plt.subplots()
no_eps, = ax.plot(all_no_eps[:,1], label='No eps')
theta_2, = ax.plot(all_theta_2[:,1], label='Theta/2')
theta_3, = ax.plot(all_theta_3[:,1], label='Theta/3')

ax.legend([no_eps, theta_2, theta_3], [r"$\epsilon = 0$", r"$\epsilon = \theta_r/2$", r"$\epsilon = \theta_r/3$"])
plt.title("Map 3")
plt.xlabel(r"Algorithm iterations $N_{iter}$")
plt.ylabel(r"Acceptance index $\omega$")
plt.xticks(range(1,21))
plt.show()