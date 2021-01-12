
from model_eqCSC import equilibrateCSC
from model_ODErates import calculateRates, takeClosest
from model_classes import *
from math import exp, log, log10
import numpy as np


def runModel(inputs):
    """
    This is the top level function to run the model. The model inputs can be
    changed by creating an instance of the ModelInputs() class then changing
    the values as desired, i.e.:
        my_inputs = ModelInputs()
        my_inputs.Q = 0.8

    Inputs:

    Returns:
    """

    #sv = State Vector, used to update the changing conditions of the system
    #sv = []

    inputs.initializeParameters() #Some inputs need additional calculations
    inputs.initializeGuesses()

    #Initialize important objects for model run
    sv = StateVector()
    output = OutputStructure()

    sv = getInitialValues(inputs,sv) #Assign the starting values to state vector

    while 1:

        inputs.updateInputsForTime(sv.time)
        sv = equilibrateCSC(inputs,sv) #Equilibrate system chemistry
        Co_rates, Ao_rates = calculateRates(inputs,sv) #Calculate rates of change of the system (terms in ODE's)
        sv = calculateTstepCSC(inputs,sv,Co_rates,Ao_rates) #Calculate time step for CSC
        output.updateCSC(sv,Co_rates,Ao_rates) #Record CSC variables

        # sv,rec_IAF    = advanceIAF(inputs,sv) #Advance Ice Albedo Feedback
            #record IAF output inside function

        sv = advanceCSC(inputs,sv, Co_rates, Ao_rates) #Update Co, Ao in CSC

        exit_flag = validateRun(sv,inputs,output)

        if exit_flag:
            output.exit_flag = exit_flag
            break

    return output

def getInitialValues(inputs,sv):

    sv.Co   = inputs.initial_Co
    sv.Ao   = inputs.initial_Ao
    sv.pCO2 = inputs.initial_pCO2
    sv.time = inputs.start_time

    return sv

def calculateTstepCSC(inputs, sv, Co_rates, Ao_rates):
    x = inputs.CSC_tstep_handling
    if x == 1:
        tstep = inputs.fixed_tstep
    elif x == 2:
        tstep_Co = (inputs.Co_threshold * sv.Co) / abs(sum(Co_rates))
        tstep_Ao = (inputs.Ao_threshold * sv.Ao) / abs(sum(Ao_rates))
        tstep = min(tstep_Co, tstep_Ao)

        tstep = min(inputs.max_tstep, tstep)
        tstep = max(inputs.min_tstep, tstep)
    else:
        raise Exception('Invalid CSC_tstep_handling')

    sv.tstepCSC = tstep
    return sv

def advanceCSC(inputs,sv,Co_rates,Ao_rates):
    sv.Co = sv.Co + sum(Co_rates)*sv.tstepCSC
    sv.Ao = sv.Ao + sum(Ao_rates)*sv.tstepCSC
    sv.time = sv.time + sv.tstepCSC
    return sv

def validateRun(sv,inputs,output):

    exit_flag = []

    error = sv.checkForNegatives()
    if error:
        exit_flag.append(999)

    #Check if past end time
    if sv.time >= inputs.end_time:
        exit_flag.append(1)

    #Check for end conditions
    x = inputs.run_mode
    if x == 2:
        #Checking for steady state if run_mode == 2
        if sv.time >= inputs.ss_time:

            t_cutoff = sv.time - inputs.ss_time #Oldest year to include in average
            t_cutoff_idx = output.time.index(takeClosest(output.time, t_cutoff))

            moving_avg = np.sum(np.multiply(output.pCO2[t_cutoff_idx:-1],output.tstepCSC[t_cutoff_idx:-1]))/inputs.ss_time
            pc_now = sv.pCO2
            pct_diff = abs(moving_avg - pc_now) / moving_avg

            steady_state_reached = (pct_diff < inputs.ss_threshold)
            if steady_state_reached:
                exit_flag.append(2)

    return exit_flag