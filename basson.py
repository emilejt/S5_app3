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

    # Normalisation des frequences de coupures pour utiliser firwin
    nyquist = 0.5 * sample_rate
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist

    # conception du filtre avec fenetre de hamming
    bandstop_filter = firwin(order + 1, [low, high], pass_zero=True, window='hamming')

    return bandstop_filter

# Fonction qui applique filtre aux signaux corrompus
def apply_filter(signal, filter_coefficient):
    filtered_signal = lfilter(filter_coefficient, 1.0, signal)
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

    sample_rate, signal = read_audio_file('note_basson_plus_sinus_1000_hz.wav')

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
    plt.show()

