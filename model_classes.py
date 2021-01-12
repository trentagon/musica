
import numpy as np
from model_eqCSC import deepOceanTemp, poreSpaceTemp, equilibriumRateConstants, carbonateSolubility
from math import exp, log, log10
from model_ODErates import takeClosest
from bisect import bisect_left

class ModelInputs:
    """
    This class represents the structure for changing model parameters.
    """

    def __init__(self):
        #Run mode
        self.run_mode = 2 #1: basic set time domain. #2: to steady state. #3: start at modern Earth and go backward
        self.start_time = 0
        self.end_time = 10**10
        self.ss_threshold = 0.01
        self.ss_time = 10**8

        #Initial Guesses
        self.seed_guess   = 0 #0: initialize normally. 1: use previous solution with same luminosity
        self.initial_Co   = 0
        self.initial_Ao   = 0
        self.initial_pCO2 = 0

        #Land setup + albedo grid handling
        self.f_bio  = 1.0  # biological weathering fraction
        self.f_land = 1.0  # land fraction compared to modern Earth

        #Climate Model
        self.lum       = 1.0   # luminosity compared to the modern Earth
        self.grad      = 1.075 # temperature gradient from surface to ocean depth
        self.Q         = 1.0   # internal heat flow compared to modern
        self.K         = 77.8  # conductivity of the sediments [m K-1]
        self.sed_depth = 1.0  # sediment thickness relative to modern Earth

        #Chemistry
        self.Mo = 1.35E21  # ocean mass [kg]
        self.ca = 0.01 # modern Ca abundance [mol kg-1]
        self.s  = 1.8E20/self.Mo #correction factor for mass balance. 1.8E20 = mass of the atmosphere

        #CSC Rate Calculations
        self.Fmod_out  = 6.0E12  # modern outgassing rate [mol C yr-1]
        self.Fmod_diss = 0.45E12  # modern seafloor dissolution rate [mol C yr-1]
        self.diss_x    = 1.0  # modern seafloor dissolution relative to prec.
        self.pCO2_mod  = 0.000280  # pCO2 on modern Earth [bar]
        self.pH_mod    = 8.2  # pH of modern ocean
        self.Fmod_carb = 10.0E12  # modern carbonate weathering rate [mol C yr-1]
        self.carb_n    = 1.75  # carbonate precipitation coefficient
        self.gamma     = 0.2  # pH dependence of seafloor weathering
        self.E_bas     = 90000.0  # temp dependence of seafloor weathering [J mol-1]
        self.CO2_xi    = 0.3  # xi term in equation S2
        self.out_m     = 1.5  # outgassing exponent
        self.Ts_mod    = 285.0  # modern (preindustrial) surface temp [K]
        self.Te        = 25.0  # e-folding temp in equations 1, S2 [K]
        self.CO2_alpha = 0.3  # alpha term in equation 1
        self.beta      = 0.1  # spreading rate dependence
        self.Hmod_mol  = 10.0 ** (-self.pH_mod)  # equation S16, initial H conc.
        self.Pmod_pore = 0.45E12  # modern pore space precipitation [mol C yr-1]

        #CSC timestep calculation
        self.CSC_tstep_handling = 1 # 1: fixed. 2: Ao,Co threshold.
        self.fixed_tstep  = 10**3
        self.Ao_threshold = 0.0001
        self.Co_threshold = 0.0001
        self.max_tstep    = 10**5
        self.min_tstep    = 10**3

        #Properties that need initialization
        self.K_diss   = 0
        self.K_ocean  = 0
        self.K_pore   = 0
        self.Fmod_sil = 0
        self.Co_0     = 0
        self.Ao_0     = 0

        #Update parameters through time
        self.getAtTime_f_bio = lambda t: self.f_bio
        self.getAtTime_f_land = lambda t: self.f_land
        self.getAtTime_CO2_alpha = lambda t: self.CO2_alpha
        self.getAtTime_CO2_xi = lambda t: self.CO2_xi
        self.getAtTime_Te = lambda t: self.Te
        self.getAtTime_carb_n = lambda t: self.carb_n
        self.getAtTime_diss_x = lambda t: self.diss_x
        self.getAtTime_grad = lambda t: self.grad
        self.getAtTime_gamma = lambda t: self.gamma
        self.getAtTime_E_bas = lambda t: self.E_bas
        self.getAtTime_beta = lambda t: self.beta
        self.getAtTime_out_m = lambda t: self.out_m
        self.getAtTime_sed_depth = lambda t: self.sed_depth
        self.getAtTime_Q = lambda t: self.Q
        self.getAtTime_K = lambda t: self.K
        self.getAtTime_lum = lambda t: self.lum


    def initializeParameters(self):
        """
            Several parameters need to be initialized. Here we use the (assumed) values
            for modern seafloor carbonate precipitation, outgassing, and the ratio of
            seafloor dissolution to carbonate precipitation to calculate the rate
            constants needed by the model.
            """

        T_s = self.Ts_mod # we initialize from the modern Earth
        T_do   = deepOceanTemp(T_s, self.grad)
        T_pore = poreSpaceTemp(T_do, self.Q, self.sed_depth, self.K)

        [K1, K2, H_CO2] = equilibriumRateConstants(T_s)

        partition = self.Fmod_diss / self.Fmod_out
        Fmod_diss = partition * self.Fmod_out * self.diss_x
        self.Fmod_sil = (1.0 - partition) * self.Fmod_out + \
                   (1 - self.diss_x) * partition * self.Fmod_out
        Pmod_pore = partition * self.Fmod_out

        # initial conditions for atmosphere-ocean system (eqns S12 to S14)
        CO2aq_o = H_CO2 * self.pCO2_mod
        HCO3_o = K1 * CO2aq_o / (10 ** -self.pH_mod)
        CO3_o = K2 * HCO3_o / (10 ** -self.pH_mod)
        DIC_o = CO3_o + HCO3_o + CO2aq_o  # total dissolved inorganic carbon
        ALK_o = 2.0*CO3_o + HCO3_o  # carbonate alkalinity

        # assume steady state for modern, so ocean precip is equal to inputs minus pore
        Pmod_ocean = self.Fmod_out + self.Fmod_carb - Pmod_pore

        omega_o = self.ca * CO3_o / carbonateSolubility(T_do)

        omega_p = self.ca * CO3_o / carbonateSolubility(T_pore)

        self.K_pore = Pmod_pore / (omega_p - 1.0) ** self.carb_n  # for pore precip.
        self.K_ocean = Pmod_ocean / (omega_o - 1.0) ** self.carb_n  # for ocean precip.

        self.K_diss = Fmod_diss / (2.88 * 10 ** -14 * 10 ** (-self.gamma * self.pH_mod) *
                              exp(-self.E_bas / (8.314 * T_pore)))

        self.Co_0 = DIC_o + self.pCO2_mod * self.s
        self.Ao_0 = ALK_o

        return

    def initializeGuesses(self):

        if all([self.initial_Co,self.initial_Ao,self.initial_pCO2]):
            #If an initial guess is provided, use it
            return

        if self.seed_guess:
            # Use an initial guess based off of previous solutions

            # Guide file must be sorted!
            filename = 'fig6_249planets.csv'
            guide = np.loadtxt(filename,delimiter=',')

            match_tuple = np.where(guide == takeClosest(guide[:,0], self.lum))
            idx_array = match_tuple[0]
            guess_idx = idx_array[0]

            self.initial_Co   = guide[guess_idx,3]
            self.initial_Ao   = guide[guess_idx, 4]
            self.initial_pCO2 = guide[guess_idx, 1]

        else:
            self.initial_Co = self.Co_0
            self.initial_Ao = self.Ao_0
            self.initial_pCO2 = self.pCO2_mod

        return

    def updateInputsForTime(self, t):
        self.f_bio = self.getAtTime_f_bio(t)
        self.f_land = self.getAtTime_f_land(t)
        self.CO2_alpha = self.getAtTime_CO2_alpha(t)
        self.CO2_xi = self.getAtTime_CO2_xi(t)
        self.Te = self.getAtTime_Te(t)
        self.carb_n = self.getAtTime_carb_n(t)
        self.diss_x = self.getAtTime_diss_x(t)
        self.grad = self.getAtTime_grad(t)
        self.gamma = self.getAtTime_gamma(t)
        self.E_bas = self.getAtTime_E_bas(t)
        self.beta = self.getAtTime_beta(t)
        self.out_m = self.getAtTime_out_m(t)
        self.sed_depth = self.getAtTime_sed_depth(t)
        self.Q = self.getAtTime_Q(t)
        self.K = self.getAtTime_K(t)
        self.lum = self.getAtTime_lum(t)
        return


class OutputStructure:
    """
    This class represents the structure for the data produced by this model for analysis and plotting.
    """
    def __init__(self):
        #All vectors containing state variables MUST be the same length

        # Primary Variables
        self.Co = []
        self.Ao = []
        self.pCO2 = []
        self.T_surf = []
        self.time = []


        # Secondary Variables
        self.pH = []
        self.omega = []
        self.T_pore = []
        self.T_deep = []
        self.tstepCSC = []

        # Rates
        self.Co_rates = []
        self.Ao_rates = []

        self.albedo_grid = []

        #Technical
        self.exit_flag = 0

    def updateCSC(self,sv,Co_rates,Ao_rates):
        # Primary Variables
        self.Co.append(sv.Co)
        self.Ao.append(sv.Ao)
        self.pCO2.append(sv.pCO2)
        self.T_surf.append(sv.T_surf)
        self.time.append(sv.time)

        # Secondary Variables
        self.pH.append(sv.pH)
        self.omega.append(sv.omega)
        self.T_pore.append(sv.T_pore)
        self.T_deep.append(sv.T_deep)
        self.tstepCSC.append(sv.tstepCSC)

        #Rates
        self.Co_rates.append(Co_rates)
        self.Ao_rates.append(Ao_rates)

        self.albedo_grid.append(sv.albedo_grid)

    def checkVectLengths(self):
        result = 1
        return result


class StateVector:
    """
    This class represents the structure for the state vector used inside the model.
    """

    def __init__(self):

        #Primary Variables
        self.Co     = 0.0
        self.Ao     = 0.0
        self.pCO2   = 0.0
        self.T_surf = 0.0
        self.time   = 0.0

        #Secondary Variables
        self.pH     = 0.0
        self.omega  = 0.0
        self.T_pore = 0.0
        self.T_deep = 0.0
        self.tstepCSC = 0.0

        self.albedo_grid = []

    def checkForNegatives(self):
        result = 0
        attributes = [self.Co,self.Ao,self.pCO2,self.T_surf,self.time,self.pH,self.omega,self.T_pore,self.T_deep,self.tstepCSC]
        for att in attributes:
            if att < 0:
                result = 1
        return result