
### File containing all relevant functions for calculating equilibrateCSC ###
import numpy as np
from math import exp, log, log10

def equilibrateCSC(inputs, sv, T_threshold = 1.0E-5, max_itter = 100):
    """
        This function will calculate the equilibrium of pCO2 and surface temperature.
        After finding this equilibrium, it will also calculate other aspects of the chemistry
        that are important for other calculations (e.g. weathering rates).

        Inputs:
            sv - state vector
            inputs - model inputs

        Returns:
            updated state vector with CSC equilibrated variables
        """

    #Unpack StateVector and calculate DIC
    pCO2 = sv.pCO2
    DIC  = sv.Co - inputs.s * sv.pCO2

    # Iterate between pCO2 and T_surf until equilibrium is reached
    T_surf_old = 0.1
    count = 0
    while 1:
        T_surf = surfaceTemp(pCO2, inputs.lum)
        T_deep = deepOceanTemp(T_surf, inputs.grad)
        T_pore = poreSpaceTemp(T_deep, inputs.Q, inputs.sed_depth, inputs.K)

        omega, pCO2, pH = equilibriumChemistry(T_surf, sv.Ao, DIC, inputs.s, inputs.initial_Ao, inputs.ca)

        count += 1
        if abs(T_surf_old - T_surf)/T_surf_old < T_threshold or count >= max_itter:
            break
        T_surf_old = T_surf

    #Update StateVector
    sv.pCO2   = pCO2
    sv.pH     = pH
    sv.T_surf = T_surf
    sv.T_deep = T_deep
    sv.T_pore = T_pore
    sv.omega  = omega
    return sv


def surfaceTemp(pCO2, flux):
    """
    This function will return the surface temperature of an Earth-like planet for
    the given partial pressure of CO2 and incident flux (normalized to modern
    Earth's). The function is defined for CO2 levels between >1.0E-7 and <10
    bar. The flux definitions are from 1.05 to 0.31 (the HZ for a Sun-like
    star). The fit is to a 4th order polynomial over CO2 and flux.

    Inputs:
        pCO2 - the CO2 partial pressure of the atmosphere [bar]
        flux - the incident flux normalised to the modern Earths (i.e. divided
               by ~1360 [W m-2])

    Returns:
        the surface temperature of the planet [K]
    """
    #the fit was done in log space for CO2
    x = np.log(pCO2)
    y = flux

    coeffs = np.array([4.8092693271e+00, -2.2201836059e+02, -6.8437057004e+01,
        -6.7369814833e+00, -2.0576569974e-01, 1.4144615786e+03,
        4.4638645525e+02, 4.4412679359e+01, 1.3641352778e+00, -2.9643244170e+03,
        -9.7844390774e+02, -9.8858815404e+01, -3.0586461777e+00,
        2.6547903068e+03, 9.0749599550e+02, 9.2870700889e+01, 2.8915352308e+00,
        -8.6843290311e+02, -3.0464088878e+02, -3.1476199768e+01,
        -9.8478712084e-01, 1.0454688611e+03, -1.4964888001e+03,
        1.0637917601e+03, -2.8114373919e+02])

    p4_in = np.array([1, x, x**2, x**3, x**4, x*y, x**2*y, x**3*y, x**4*y,
        x*y**2, x**2*y**2, x**3*y**2, x**4*y**2, x*y**3, x**2*y**3, x**3*y**3,
        x**4*y**3, x*y**4, x**2*y**4, x**3*y**4, x**4*y**4, y, y**2, y**3,
        y**4])


    return np.sum(p4_in*coeffs)


def deepOceanTemp(Ts, gradient, min_temp=271.15):
    """
    Determine the deep ocean temperature based on the surface temperature. The
    intercept term is chosen so that gradient*Ts+intercept gives the correct
    surface temperature. In the case of the modern Earth, that would be the
    modern average surface temperature. This function corresponds to equation
    S20.

    Inputs:
        Ts        - surface temperature [K]
        gradient  - total temperature gradient in the ocean [dimensionless]
        min_temp  - the minimum allowable temperature at the bottom of the
                    ocean. For an Earth-like planet below 271.15 K (the default
                    value) the ocean would freeze.

    Returns:
        Td - the temperature at the bottom of the ocean [K]
    """

    # intercept chosen to reproduce initial (modern) temperature
    intercept = 274.037 - gradient*Ts
    Td = np.max([np.min([gradient*Ts+intercept, Ts]), min_temp])

    return Td


def poreSpaceTemp(Td, Q, S_thick, K):
    """
    This function will calculate the temperature of the pore space. This is
    based on equation S4.

    Inputs:
        T_D     - deep ocean temperature [K]
        Q       - pore space heat flow relative to modern Earth [dimensionless]
        S_thick - the thickness of the ocean sediment relative to modern Earth
                  [dimensionless]
        K       - conductivity of the pore space sediments [m K-1]

    Returns:
    T_pore - the temperature of the pore space [K]
    """

    sed = S_thick*700 #modern Earth sediment thickness is ~700 m
    T_pore = Td + Q*sed/K

    return T_pore


def equilibriumChemistry(T, alk, carb, s, alk_init, Ca_init):
    """
    Calculate the carbonate equilibrium and alkalinity. This can be used for
    either the atmosphere-ocean or the pore space. This function represents
    equations S11-S18.

    Inputs:
        T        - the temperature of the system [K]
        alk      - the alkalinity of the system [mol eq]
        carb     - the carbon abundance in the system [mol]
        s        - correction factor for mass balance
        alk_init - initial alkalinity of the system [mol eq]
        Ca_init  - initial calcium ion concentration in the system [mol]

    Returns:
        omega - the saturation state of the system
        pCO2  - the partial pressure of CO2 [bar]
        pH    - pH of the system
    """

    # get the rate constants and Henry's constant

    [K1, K2, H_CO2] = equilibriumRateConstants(T)

    # use equation S15 to first calculate the H+ ion concentration
    roots = np.roots([alk / (K1 * K2) * (1.0 + s / H_CO2),
                      (alk - carb) / K2,
                      alk - 2.0 * carb])

    H_ion = np.max(roots)  # just take the positive root
    pH = -log10(H_ion)  # equation S16 (aka pH definition)

    CO3 = alk / (2.0 + H_ion / K2)  # S14 with S11
    HCO3 = alk - 2.0 * CO3  # S11
    CO2_aq = H_ion * HCO3 / K1  # S13
    pCO2 = CO2_aq / H_CO2  # S12
    Ca_ion = 0.5 * (alk - alk_init) + Ca_init  # S17
    K_sp = carbonateSolubility(T)
    omega = Ca_ion * CO3 / K_sp  # S18

    return [omega, pCO2, pH]


def equilibriumRateConstants(T):
    """
    Calculates the carbon chemistry equilibrium constants as a function of
    temperature following the method in Appendix A of JKT 2018 (you actually
    have to look at their 2017 paper for these equations).

    Inputs:
        T - the temperature of the system [K]

    Returns:
        K1    - the first apparent dissociation rate constant of carbonic acid
        K2    - the second apparent dissociation rate constant of carbonic acid
        H_CO2 - Henry's law constant for CO2
    """

    pK1 = 17.788 - .073104 * T - .0051087 * 35 + 1.1463 * 10 ** -4 * T ** 2
    pK2 = 20.919 - .064209 * T - .011887 * 35 + 8.7313 * 10 ** -5 * T ** 2
    H_CO2 = exp(9345.17 / T - 167.8108 + 23.3585 * log(T) +
                (.023517 - 2.3656 * 10 ** -4 * T + 4.7036 * 10 ** -7 * T ** 2) * 35)

    K1 = 10.0 ** -pK1
    K2 = 10.0 ** -pK2

    return [K1, K2, H_CO2]

def carbonateSolubility(T):
    """
    Calculates carbonate solubility rate constant as a function of temperature.
    See Appendix A of JKT 2018 for further details (you'll need to look at
    their 2017 paper for these actual equations - but Appendix A tells you
    that).

    Inputs:
        T - the temperature of the system [K]

    Returns:
        result - the solubility rate constant
    """
    bo = -0.77712
    b1 = 0.0028426
    b2 = 178.34
    co = -0.07711
    do = 0.0041249
    S = 35.0
    logK0=-171.9065-0.077993*T+2839.319/T+71.595*log10(T)
    logK=logK0+(bo+b1*T+b2/T)*S**0.5+co*S+do*S**1.5

    result = 10.0**logK

    return result
