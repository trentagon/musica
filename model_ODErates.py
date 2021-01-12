
### File containing all relevant functions for calculating equilibrateCSC ###
import numpy as np
from math import exp, log, log10
from bisect import bisect_left


def calculateRates(inputs,sv):

        #only precipitate when omega > 1
        if sv.omega > 1:
            P_ocean = inputs.K_ocean*inputs.f_land*(sv.omega - 1.0)**inputs.carb_n #S19
            P_pore = inputs.K_pore*(sv.omega - 1.0)**inputs.carb_n #S19
        else:
            P_ocean = 0
            P_pore  = 0

        # calculate all the parameters needed
        F_out = globalOutgassing(inputs.Fmod_out, inputs.Q, inputs.out_m)


        deltaTs = sv.T_surf - inputs.Ts_mod
        F_carb = continentalCarbonateWeathering(inputs.f_bio, inputs.f_land,
                                                inputs.Fmod_carb, sv.pCO2, inputs.pCO2_mod, inputs.CO2_xi,
                                                deltaTs, inputs.Te)

        F_sil = continentalSilicateWeathering(inputs.f_bio, inputs.f_land,
                                              inputs.Fmod_sil, sv.pCO2, inputs.pCO2_mod, inputs.CO2_alpha, deltaTs,
                                              inputs.Te)

        r_sr = spreadingRate(inputs.Q, inputs.beta)
        H_mol = 10 ** (-sv.pH)
        F_diss = seafloorBasaltDissolution(inputs.K_diss, r_sr, inputs.E_bas,
                                           sv.T_pore, H_mol, inputs.Hmod_mol, inputs.gamma)


        #Helpful Definitions:
        #Mo      - mass of the ocean [kg]
        #F_out   - global outgassing flux [mol C yr-1]
        #F_carb  - continental carbonate weathering rate [mol C yr-1]
        #P_ocean - precipitation flux of carbonates in the ocean [mol C yr-1]
        #F_sil   - continental silicate weathering flux [ mol C yr-1]
        #F_diss  - seafloor weathering from basalt dissolution [mol eq yr-1]
        #P_pore  - carbonate precipitation flux in the pore space [mol C yr-1]

        #dCo_dt - change in atmosphere-ocean carbon concentration [mol C yr-1]
        #dAo_dt - change in atmosphere-ocean alkalinity [mol eq yr-1]

        #dCo_dt = (F_out + F_carb - P_ocean - P_pore) / Mo
        #dAo_dt = 2 * (F_sil + F_carb - P_ocean + F_diss - P_pore) / Mo


        Co_rates = np.divide([F_out, F_carb, -P_ocean, -P_pore], inputs.Mo)
        Ao_rates = 2*np.divide([F_sil, F_carb, -P_ocean, F_diss, -P_pore],inputs.Mo)
        #Divide  by Mo/2 because there is a factor of 2 in the original ODE

        return Co_rates, Ao_rates


def globalOutgassing(Fmod_out, Q, m):
    """
    This function will calculate the outgassing flux (equation S9).

    Inputs:
        Fmod_out - the modern Earth's outgassing rate [mol C yr-1]
        Q        - pore space heat flow relative to modern Earth [dimensionless]
        m        - scaling parameter [dimensionless]

    Returns:
        F_out - the global outgassing flux [mol C yr-1]
    """

    F_out = Fmod_out * Q ** m

    return F_out

def continentalCarbonateWeathering(f_bio, f_land, Fmod_carb, pCO2, pCO2mod,
        eps, deltaTs, Te):
    """
    The rate of carbon liberated by continental weathering will be returned from
    this function. This is function S2 in JKT.

    Inputs:
        f_bio     - biological enhancement of weathering, set to 1 for the
                    modern Earth [dimensionless]
        f_land    - land fraction compared to the modern Earth [dimensionless]
        Fmod_carb - Earth's modern carbonate weathering rate [mol yr-1]
        pCO2      - partial pressure of CO2 [Pa]
        pCO2mod   - Earth's modern (preindustrial) CO2 partial pressure [Pa]
        eps       - empirical constant [dimensionless]
        deltaTs   - difference in global mean surface temperature [K]
        Te        - defines temperature dependence of weathering [K]

    Returns:
        F_carb - the carbonate weathering rate [mol yr-1]
    """

    F_carb = f_bio*f_land*Fmod_carb*(pCO2/pCO2mod)**eps
    if Te > 0:
        F_carb = f_bio*f_land*Fmod_carb*(pCO2/pCO2mod)**eps*exp(deltaTs/Te)

    return F_carb

def continentalSilicateWeathering(f_bio, f_land, Fmod_sil, pCO2, pCO2mod,
        alpha, deltaTs, Te):
    """
    The rate of silicate weathering  from continents. This function corresponds
    to equation 1 from JKT.

    Inputs:
        f_bio    - biological enhancement of weathering, set to 1 for the
                   modern Earth [dimensionless]
        f_land   - land fraction compared to the modern Earth [dimensionless]
        Fmod_sil - Earth's modern silicate weathering rate [mol yr-1]
        pCO2     - partial pressure of CO2 [Pa]
        pCO2mod  - Earth's modern (preindustrial) CO2 partial pressure [Pa]
        alpha    - empirical constant [dimensionless]
        deltaTs  - difference in global mean surface temperature [K]
        Te       - defines temperature dependence of weathering [K]

    Returns:
        F_sil - the silicate weathering rate [mol yr-1]
    """

    F_sil = f_bio*f_land*Fmod_sil*(pCO2/pCO2mod)**alpha
    if Te > 0:
        F_sil = f_bio*f_land*Fmod_sil*(pCO2/pCO2mod)**alpha*exp(deltaTs/Te)

    return F_sil

def spreadingRate(Q, beta):
    """
    Calculates the spreading rate on the planet.

    Inputs:
        Q    - pore space heat flow relative to modern Earth [dimensionless]
        beta - scaling parameter [dimensionless]

    Returns:
        r_sr - the spreading rate relative to the modern Earth [dimensionless]
    """

    r_sr = Q**beta

    return r_sr

def seafloorBasaltDissolution(k_diss, r_sr, E_bas, T_pore, H_mol, Hmod_mol,
        gamma):
    """
    This function will calculate the rate of basalt dissolution on the
    seafloor. This function represents equation S3 of JKT.

    Inputs:
        k_diss   - proportionality constant chosen to match modern flux
                   [dimensionless]
        r_sr     - spreading rate compared to modern [dimensionless]
        E_bas    - effective activation energy of dissolution [J mol-1]
        T_pore   - temperature of the pore space [K]
        H_mol    - hydrogen ion molality in the pore space [mol kg-1]
        Hmod_mol - the modern H ion molality in pre space [mol kg-1]
        gamma    - empirical scaling parameter [dimensionless]

    Returns:
        F_diss - rate of seafloor basalt dissolution [mol eq yr-1]
    """

    Rg = 8.314 #universal gas constant [J mol-1 K-1]
    F_diss = k_diss*r_sr*exp(-E_bas/(Rg*T_pore))*2.88*10**-14*\
            10**(-gamma*(-log10(H_mol))) #see code from JKT

    return F_diss

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before
