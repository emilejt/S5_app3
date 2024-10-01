import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, freqz

# Fonction pour lecture du fichier wav qui retourne le signal et sa frequence d'echantillonage
def read_audio_file(filename):
    sample_rate, signal = wavfile.read(filename)
    return sample_rate, signal

# Fonction qui determine les coefficient du filtre
def design_bandstop_filter(sample_rate, cutoff_low, cutoff_high, order):
    freq_central = (cutoff_high + cutoff_low) / 2
    fc_passe_bas = cutoff_high - freq_central


    w = (2* np.pi * freq_central) / sample_rate

    m = (fc_passe_bas * order) / sample_rate
    k = (2 * m) + 1

    # Sequence d'echantillons autour de 0
    data = np.linspace(-(order / 2) + 1, order / 2, order)
    dn = [1 if data[i] == 0 else 0 for i in range(0, order, 1)]
    h_filtre = []

    for n in data:
        # pour eviter la division par 0
        if n == 0:
            dn.append(1)
            h_filtre.append(k / order)
        else:
            dn.append(0)
            h_filtre.append((np.sin((np.pi * n * k) / order) / np.sin((np.pi * n) / order )) / order)
    
    # Applique fenetre de Hamming pour ameliorer filtre
    window = np.hamming(order)
    h_filtre = h_filtre * window

    # transformation en filtre coupe-bande avec sinusoidale centre sur freq_central
    h_cb = [dn[i] - np.multiply(2 * h_filtre[i], np.cos(w * data[i])) for i in range(0, order, 1)]

    return h_cb

# Fonction qui applique filtre aux signaux corrompus
def apply_filter(signal, filter_coefficient):
    filtered_signal = np.convolve(signal, filter_coefficient, mode='same')
    return filtered_signal

# Fonction qui cree fichier wav avec nom au choix
def save_audio_file(filename, sample_rate, signal):
    signal_normalized = signal / np.max(np.abs(signal))
    signal_int16 = np.int16(signal_normalized * 32767)
    wavfile.write(filename, sample_rate, signal_int16)

if __name__ == "__main__":
    cutoff_low = 960
    cutoff_high = 1040
    order = 6000

    # Lecture du fichier wav corrompue
    sample_rate, signal = read_audio_file('note_basson_plus_sinus_1000_hz.wav')

    # Creation et application du filtre
    bandstop_filter = design_bandstop_filter(sample_rate, cutoff_low, cutoff_high, order)
    filtered_signal = apply_filter(signal, bandstop_filter)

    save_audio_file('clean_signal.wav', sample_rate, filtered_signal)

    w, h = freqz(bandstop_filter, worN=8000)
    plt.plot(0.5 * sample_rate * w / np.pi, np.abs(h), 'b')
    plt.title("Réponse en fréquence du filtre coupe-bande")
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Gain')
    plt.grid()
    plt.axvline(cutoff_low, color='red', linestyle='--')
    plt.axvline(cutoff_high, color='red', linestyle='--')
    plt.xlim([0, 3000])
    plt.show()

