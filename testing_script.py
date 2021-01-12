
from result_analysis import plot_results
from model_main import *
from model_classes import *
from model_eqCSC import *
import numpy as np
from math import exp, log, log10
import time
import random


inputs = ModelInputs()
inputs.lum = 1
inputs.run_mode = 2 #Run for steady state

# inputs.initial_Co = 0.0017168419854444441
# inputs.initial_Ao = 0.0017697155934936067
# inputs.initial_pCO2 = 0.000287458355454603

tic = time.perf_counter()
output = runModel(inputs)
toc = time.perf_counter()

import matplotlib.pyplot as plt

# plt.plot(output.time,output.T_surf)
# plt.plot(output.time,output.pH)
# plt.semilogy(output.time,output.pCO2)

print("Finished in %0.2f seconds with exit flag %0.0f (length=%0.0f)"%(toc-tic,output.exit_flag[0],len(output.exit_flag)))
print("pH = %0.2e"%(output.pH[-1]))
print("pCO2 = %2.3e bar"%(output.pCO2[-1]))
print("Surface temperature = %0.2f K"%(output.T_surf[-1]))
print("Model time = %1.3e yrs"%(output.time[-1]))
print("Mean tstep = %0.2f years"%(np.mean(output.tstepCSC)))

# print('Co: Initial %2.3e | Final %2.3e'%(inputs.initial_Co,output.Co[-1]))
# print('Ao: Initial %2.3e | Final %2.3e'%(inputs.initial_Ao,output.Ao[-1]))
# print('pCO2: Initial %2.3e | Final %2.3e'%(inputs.initial_pCO2,output.pCO2[-1]))

#plot_results(output, time=[0,5E6])
# import matplotlib.pyplot as plt
# plt.plot(output.time,output.pCO2)

# lehmer_co2 = 2.875e-04
# print(surfaceTemp(lehmer_co2, inputs.lum))
# print(surfaceTemp(output.pCO2[-1], inputs.lum))