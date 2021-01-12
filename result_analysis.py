
def plot_results(output, time=[0,-1]):

    import matplotlib.pyplot as plt
    import numpy as np
    from model_ODErates import takeClosest



    if time != [0,-1]:
        if time[0] == 0:
            finish_match_tuple = np.where(output.time == takeClosest(output.time, time[1]))
            finish_idx_array = finish_match_tuple[0]
            s = 0 # Start index
            f = finish_idx_array[0]  # Finish index
        else:
            start_match_tuple = np.where(output.time == takeClosest(output.time, time[0]))
            finish_match_tuple = np.where(output.time == takeClosest(output.time, time[1]))

            start_idx_array = start_match_tuple[0]
            finish_idx_array = finish_match_tuple[0]

            s = start_idx_array[0] # Start index
            f = finish_idx_array[0] # Finish index
    else:
        s = time[0]  # Start index
        f = time[1]  # Finish index


    t = np.array(output.time[s:f]) / 10 ** 9

    co_labels = ["Outgassing", "Continental Carbonate Weathering", "Carbonate Precipitation in Ocean",
                 "Carbonate Precipitation in Pore"]
    ao_labels = ["Continental Silicate Weathering", "Continental Carbonate Weathering",
                 "Carbonate Precipitation in Ocean", "Seafloor Weathering from Basalt Dissolution",
                 "Carbonate Precipitation in Pore"]

    fig = plt.figure(figsize=[15, 10])

    ax1 = fig.add_subplot(331)
    ax1.plot(t, output.Co[s:f])
    ax1.set_ylabel("Concentration [mol C kg^-1]")
    ax1.set_xlabel("Time [Ga]")
    ax1.set_title("Co")

    ax2 = fig.add_subplot(335)
    ax2.plot(t, output.T_surf[s:f], label="Surface")
    ax2.plot(t, output.T_deep[s:f], label="Deep")
    ax2.plot(t, output.T_pore[s:f], label="Pore")
    plt.legend(loc='upper left', fontsize=8)
    ax2.set_ylabel("K")
    ax2.set_xlabel("Time [Ga]")
    ax2.set_title("Temperature")

    ax3 = fig.add_subplot(334)
    ax3.plot(t, output.pCO2[s:f])
    ax3.set_ylabel("bar")
    ax3.set_xlabel("Time [Ga]")
    ax3.set_title("pCO2")

    ax4 = fig.add_subplot(339)
    ax4.plot(t, output.omega[s:f])
    ax4.set_ylabel("Saturation State")
    ax4.set_xlabel("Time [Ga]")
    ax4.set_title("Omega")

    ax5 = fig.add_subplot(332)
    ax5.plot(t, output.Ao[s:f])
    ax5.set_ylabel("Alkalinity [mol eq]")
    ax5.set_xlabel("Time [Ga]")
    ax5.set_title("Ao")

    ax6 = fig.add_subplot(336)
    ax6.plot(t, output.pH[s:f])
    ax6.set_ylabel("pH")
    ax6.set_xlabel("Time [Ga]")
    ax6.set_title("pH")

    ax7 = fig.add_subplot(338)
    ao_rates = np.array(output.Ao_rates)
    ao_rates = ao_rates[s:f,:]
    for i in range(ao_rates.shape[1]):
        ax7.plot(t, ao_rates[:, i], label=ao_labels[i])

    plt.legend(loc='lower right', fontsize=8)
    ax7.set_ylabel("Rate [mol eq yr-1]")
    ax7.set_xlabel("Time [Ga]")
    ax7.set_title("Rate of change of alkalinity")

    ax8 = fig.add_subplot(337)
    co_rates = np.array(output.Co_rates)
    co_rates = co_rates[s:f,:]
    for i in range(co_rates.shape[1]):
        ax8.plot(t, co_rates[:, i], label=co_labels[i])

    plt.legend(loc='lower right', fontsize=8)
    ax8.set_ylabel("Rate [mol C yr-1]")
    ax8.set_xlabel("Time [Ga]")
    ax8.set_title("Rate of change of carbon concentration")

    ax9 = fig.add_subplot(333)
    ax9.plot(t, output.tstepCSC[s:f])
    ax9.set_ylabel("years")
    ax9.set_xlabel("Time [Ga]")
    ax9.set_title("Timestep")

    fig.tight_layout()

    return