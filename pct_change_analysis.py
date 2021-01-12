

import numpy as np
import matplotlib.pyplot as plt

f_ao = 'ao_rates_pct.csv'
f_co = 'co_rates_pct.csv'
f_t  = 'times_pct.csv'

# f_ao = 'ao_xxx.csv'
# f_co = 'co_xxx.csv'
# f_t  = 'times_xxx.csv'

ao = np.loadtxt(f_ao,delimiter=',')
co = np.loadtxt(f_co,delimiter=',')
t = np.loadtxt(f_t,delimiter=',')

co_pct = np.ndarray(shape=(co.shape[0]-1,co.shape[1]))
ao_pct = np.ndarray(shape=(ao.shape[0]-1,ao.shape[1]))

for i in range(co.shape[1]):
    for j in range(1,co.shape[0]):
        co_pct[j-1,i] = 100*abs((co[j,i] - co[j-1,i])/co[j-1,i])

for i in range(ao.shape[1]):
    for j in range(1,ao.shape[0]):
        ao_pct[j-1,i] = 100*abs((ao[j,i] - ao[j-1,i])/ao[j-1,i])


co_labels =["Outgassing","Continental Carbonate Weathering","Carbonate Precipitation in Ocean","Carbonate Precipitation in Pore"]
ao_labels =["Continental Silicate Weathering","Continental Carbonate Weathering","Carbonate Precipitation in Ocean","Seafloor Weathering from Basalt Dissolution","Carbonate Precipitation in Pore"]

fig = plt.figure(figsize=[15,10])
ax1 = fig.add_subplot(221)
for i in range(co.shape[1]):
    ax1.plot(t/10**9,co[:,i],label=co_labels[i])

plt.legend(loc='upper left', fontsize= 8)
ax1.set_ylabel("Rate [mol C yr-1]")
ax1.set_xlabel("Time [Ga]")
ax1.set_title("Rate of change of carbon concentration")


ax2 = fig.add_subplot(222)
for i in range(ao.shape[1]):
    ax2.plot(t/10**9,ao[:,i],label=ao_labels[i])

plt.legend(loc='upper left', fontsize= 8)
ax2.set_ylabel("Rate [mol eq yr-1]")
ax2.set_xlabel("Time [Ga]")
ax2.set_title("Rate of change of alkalinity")

ax3 = fig.add_subplot(223)
for i in range(co.shape[1]):
    ax3.plot(t[1:]/10**9,co_pct[:,i],label=co_labels[i])

plt.legend(loc='upper left', fontsize= 8)
ax3.set_ylabel("Percent change (%)")
ax3.set_xlabel("Time [Ga]")
ax3.set_title("Percent change of carbon concentration rates")
ax3.set_ylim([0,2])

ax4 = fig.add_subplot(224)
for i in range(ao.shape[1]):
    ax4.plot(t[1:]/10**9,ao_pct[:,i],label=ao_labels[i])

plt.legend(loc='upper left', fontsize= 8)
ax4.set_ylabel("Percent change (%)")
ax4.set_xlabel("Time [Ga]")
ax4.set_title("Percent change of alkalinity rates")
ax4.set_ylim([0,2])

plt.show()