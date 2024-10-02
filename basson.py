import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, freqz
import prob


# Fonction pour lecture du fichier wav qui retourne le signal et sa frequence d'echantillonage
def read_audio_file(filename):
    sample_rate, signal = wavfile.read(filename)
    signal_normalized = signal / np.max(np.abs(signal))
    return sample_rate, signal_normalized

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

def plot_signal_with_envelope(envelope):
    fig, ax = plt.subplots(1)

    ax.plot(envelope)
    ax.set_title("Enveloppe du signal initial")
    ax.set_xlabel("Échantillons")
    ax.set_ylabel("Amplitude")
    plt.show()

def plot_impulse_response(filter_coefficients):
    # Créer un signal d'impulsion
    impulse_length = len(filter_coefficients)
    impulse = np.zeros(impulse_length)
    impulse[impulse_length // 2] = 1  # Impulsion au centre
    
    # Réponse à l'impulsion
    response = apply_filter(impulse, filter_coefficients)

    n = np.arange(-impulse_length // 2, impulse_length // 2)  # Indices de -N/2 à N/2
    
    # Tracer la réponse à l'impulsion
    plt.figure(figsize=(10, 4))
    plt.plot(n, response)
    plt.title("Réponse à l'Impulsion h(n)")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.xlim(int(-len(filter_coefficients) / 2), int(len(filter_coefficients) / 2)) 
    plt.grid()
    plt.show()

def plot_sine_response(frequency, filter_coefficients):

    sine_wave = np.sin(2 * np.pi * frequency)  # Signal sinusoïdal de 1000 Hz
    response = apply_filter(sine_wave, filter_coefficients)

    n = np.arange(-len(filter_coefficients) // 2, len(filter_coefficients) // 2)  # Indices de -N/2 à N/2

    plt.figure(figsize=(10, 4))
    plt.plot(n, response, label='Réponse du Filtre')  # Utiliser n pour l'axe x
    plt.title("Réponse à une Sinusoïde de 1000 Hz")
    plt.xlabel("Échantillons")  # Modifier l'étiquette de l'axe des x
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim(int(-len(filter_coefficients) / 2), int(len(filter_coefficients) / 2)) 
    plt.grid()
    plt.show()

def plot_amp_phase(filter_coefficients):
    w, amp = freqz(filter_coefficients)
    angles = np.unwrap(np.angle(amp))
    fig, ax1 = plt.subplots()
    ax1.set_title('Réponse en fréquence du filtre coupe-bande')
    ax1.plot(w,20*np.log10(np.abs(amp)), color='b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Fréquence normalisé [rad/échantillon]')
    ax2 = ax1.twinx()
    ax2.plot(w, angles, 'g-', color='r')
    ax2.set_ylabel('Phase (rad)', color='r')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()


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

    # Affichage des spectres d'amplitude des signaux avant et apres filtrage
    basson_freqs, basson_amp, basson_phase, fundamental = prob.get_frequencies(signal, sample_rate, 32)
    prob.plot_spectrum(signal, sample_rate, title="Spectre d'amplitude avant filtrage")
    basson_freqs_clean, basson_amp_clean, basson_phase_clean, fundamental_clean = prob.get_frequencies(filtered_signal, sample_rate, 32)
    prob.plot_spectrum(filtered_signal, sample_rate, title="Spectre d'amplitude apres filtrage")

    enveloppe = prob.get_envelope(order, signal)
    #plot_signal_with_envelope(enveloppe)

    # Synthese du basson filtered
    duration = len(filtered_signal) / sample_rate
    reproduced_signal = prob.reproduce_signal(basson_freqs_clean, basson_amp_clean, basson_phase_clean, duration, sample_rate)
    prob.plot_spectrum(reproduced_signal, sample_rate, title="Spectre du son reproduit avec harmoniques")


    sample_rate_clean, signal_clean = read_audio_file('clean_signal.wav')
    # prob.plot_spectrum(signal_clean, sample_rate_clean, title="Spectre du son sans bruit 1000Hz", harmonics=harmonics_basson_freqs[:32])
    # passe_bas_N = prob.get_fir_N(np.pi/1000)
    # envelope = prob.get_envelope(passe_bas_N, filtered_signal)
    # prob.plot_envelope(envelope, sample_rate)
    # plot_impulse_response(bandstop_filter)

    # plot_sine_response(1000, bandstop_filter)

    # plot_amp_phase(bandstop_filter)

    # w, h = freqz(bandstop_filter, worN=8000)
    # plt.plot(0.5 * sample_rate * w / np.pi, np.abs(h), 'b')
    # plt.title("Réponse en fréquence du filtre coupe-bande")
    # plt.xlabel('Fréquence (Hz)')
    # plt.ylabel('Gain')
    # plt.grid()
    # plt.axvline(cutoff_low, color='red', linestyle='--')
    # plt.axvline(cutoff_high, color='red', linestyle='--')
    # plt.xlim([0, 3000])
    # plt.show()


    # fft_basson = np.fft.fft(signal)
    # basson_freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)
    #
    # fig, ax = plt.subplots(1)
    # ax.plot(basson_freqs, 20 * np.log10(np.abs(fft_basson)))
    # ax.set_xlim(0,1500)
    # ax.set_title("Spectres de fourier avant filtrage")
    # ax.set_xlabel("fréquence (Hz)")
    # ax.set_ylabel("Amplitude (dB)")
    #
    # fig, spec = plt.subplots(1)
    #
    # fft_basson_clean = np.fft.fft(signal_clean)
    # basson_freqs_clean = np.fft.fftfreq(len(signal_clean), d=1/sample_rate_clean)
    #
    # spec.plot(basson_freqs_clean, 20*np.log10(np.abs(fft_basson_clean)))
    # spec.set_xlim(0,1500)
    # spec.set_title("Spectre de fourier après filtrage")
    # spec.set_xlabel("fréquence (Hz)")
    # spec.set_ylabel("Amplitude (dB)")
    #
    # plt.show()


    # fft_basson = np.fft.fft(signal)
    # basson_freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

    # fig, ax = plt.subplots(1)
    # ax.plot(basson_freqs, 20 * np.log10(np.abs(fft_basson)))
    # ax.set_xlim(0,1500)
    # ax.set_title("Spectres de fourier avant filtrage")
    # ax.set_xlabel("fréquence (Hz)")
    # ax.set_ylabel("Amplitude (dB)")

    # fig, spec = plt.subplots(1)

    # fft_basson_clean = np.fft.fft(signal_clean)
    # basson_freqs_clean = np.fft.fftfreq(len(signal_clean), d=1/sample_rate_clean)

    # spec.plot(basson_freqs_clean, 20*np.log10(np.abs(fft_basson_clean)))
    # spec.set_xlim(0,1500)
    # spec.set_title("Spectre de fourier après filtrage")
    # spec.set_xlabel("fréquence (Hz)")
    # spec.set_ylabel("Amplitude (dB)")

    # plt.show()

    

