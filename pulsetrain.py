import numpy as np 
import matplotlib 
matplotlib.use("QtAgg") 
import matplotlib.pyplot as plt



# Example: results collected from CurveFit / FFT
sideband_data = {
    "SB1": {"HO": 13, "A": 1.2, "omega": 1.2, "phi": 0},
    "SB2": {"HO": 15,"A": 0.7, "omega": 1.2, "phi": 0},
    "SB3": {"HO": 17,"A": 0.4, "omega": 1.2, "phi": 0},
}

harmonics = {
        "H13": {"A": 1, "q": 13, "omega": 1.2, "phi": -1.607},
        "H15": {"A": 1, "q": 15, "omega": 1.2, "phi": 0.0},
        "H17": {"A": 1, "q": 17, "omega": 1.2, "phi": 1.287},
        "H19": {"A": 1, "q": 19, "omega": 1.2, "phi": 2.307},
}


# Time axis
t_start = 0.0
t_end = 5.5      # adjust to your pulse repetition period
num_points = 10000
t = np.linspace(t_start, t_end, num_points)





def reconstruct_pulse_train(harmonics, t):
    signal = np.zeros_like(t)
    pulse = 0

    plt.figure(figsize=(10,4))
    for h, params in harmonics.items():
        A = params["A"]
        q = params["q"]
        omega = params["omega"]
        phi = params["phi"]
        
        signal = A * np.cos(2 * q * omega * t + phi)
        
        plt.plot(t, signal, label=f"H{q}", alpha=0.5)

        pulse += signal
    
    plt.plot(t, pulse, label="pulse", lw=2, c='r')
    plt.xlabel("Time (fs)")
    plt.ylabel("Signal (a.u.)")
    plt.title("Reconstructed Pulse Train")
    plt.legend()
    plt.savefig("APT")
    plt.show()

    return signal




pulse = reconstruct_pulse_train(harmonics, t)














