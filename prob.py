from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


# Fonction pour lire un fichier audio .wav
def read_file(filename):
    sample_rate, signal = wavfile.read(filename)
    return sample_rate, signal


# Appliquer une fenêtre de Hamming au signal pour minimiser les effets de fuite spectrale
def hamming(signal):
    hamming_window = np.hamming(len(signal))
    windowed_signal = np.multiply(signal, hamming_window)
    return windowed_signal


# Extraire les fréquences, amplitudes, et phases des harmoniques principales à partir d'un signal FFT
def get_ideal_frequencies(signal, sample_rate, sinus_count):
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_signal), 1 / sample_rate)  # Obtenir les fréquences correspondantes

    # Trouver la fréquence fondamentale (plus grande amplitude)
    index_fundamental = np.argmax(np.abs(fft_signal))
    fundamental = np.abs(frequencies[index_fundamental])
    print("La# fundamental frequency: " + str(fundamental))

    sinus_freqs = []
    sinus_amplitudes = []
    sinus_phases = []

    # Stocker la fréquence fondamentale, l'amplitude et la phase
    sinus_freqs.append(fundamental)
    sinus_amplitudes.append(np.abs(fft_signal[index_fundamental]))
    sinus_phases.append(np.angle(fft_signal[index_fundamental]))

    # Extraire les harmoniques (multiples de la fréquence fondamentale)
    for i in range(2, sinus_count + 1):  # Start from the 2nd harmonic
        harmonic_freq = fundamental * i
        closest_idx = np.argmin(np.abs(frequencies - harmonic_freq))  # Find the closest frequency bin

        sinus_freqs.append(frequencies[closest_idx])
        sinus_amplitudes.append(np.abs(fft_signal[closest_idx]))
        sinus_phases.append(np.angle(fft_signal[closest_idx]))

    print("Frequencies:", sinus_freqs)
    print("Amplitudes:", sinus_amplitudes)
    print("Phases:", sinus_phases)
    return sinus_freqs, sinus_amplitudes, sinus_phases, fundamental


def get_frequencies(signal, sample_rate, sinus_count):
    # Appliquer la FFT pour obtenir le spectre de fréquence
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_signal), 1 / sample_rate)  # Obtenir les fréquences correspondantes
    magnitudes = np.abs(fft_signal)  # Obtenir les amplitudes (module de la FFT)

    # Limiter aux fréquences positives
    positive_freqs = frequencies[:len(frequencies) // 2]
    positive_magnitudes = magnitudes[:len(magnitudes) // 2]

    # Trouver la fréquence fondamentale (plus grande amplitude)
    index_fundamental = np.argmax(positive_magnitudes)
    fundamental = positive_freqs[index_fundamental]

    # Trouver les pics dans le spectre de fréquence positif
    peaks, _ = find_peaks(positive_magnitudes, height=0)

    # Trier les pics par amplitude (les plus grands d'abord) et sélectionner les 32 plus grands
    sorted_peaks = sorted(peaks, key=lambda x: positive_magnitudes[x], reverse=True)
    largest_peaks = sorted_peaks[:sinus_count]  # Prendre les 32 plus grands pics

    # Initialiser les listes pour les fréquences, amplitudes et phases
    sinus_freqs = []
    sinus_amplitudes = []
    sinus_phases = []

    # Extraire les informations des pics trouvés
    for peak_idx in largest_peaks:
        sinus_freqs.append(positive_freqs[peak_idx])  # Fréquence du pic
        sinus_amplitudes.append(positive_magnitudes[peak_idx])  # Amplitude du pic
        sinus_phases.append(np.angle(fft_signal[peak_idx]))  # Phase du pic

    # Afficher les fréquences trouvées pour validation
    print("Frequencies:", sinus_freqs)
    print("Amplitudes:", sinus_amplitudes)
    print("Phases:", sinus_phases)

    return sinus_freqs, sinus_amplitudes, sinus_phases, fundamental


# Reproduire un signal à partir de ses sinus les plus importantes
def reproduce_signal(frequencies, amplitudes, phases, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration))

    reproduced_signal = np.zeros_like(t)

    for i in range(0, len(amplitudes)-1):
        reproduced_signal += amplitudes[i - 1] * np.sin(2 * np.pi * frequencies[i] * t + phases[i - 1])

    return reproduced_signal


#cree une note a partir de sa fondamentale
def synthetize_note(fundamental, amplitudes, phases, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration))

    reproduced_signal = np.zeros_like(t)

    for i in range(1, len(amplitudes)):
        reproduced_signal += amplitudes[i - 1] * np.sin(2 * np.pi * i * fundamental * t + phases[i - 1])

    return reproduced_signal


# Calculer l'ordre N d'un filtre FIR coupe-bas pour l'enveloppe temporelle
def get_fir_N(cutoff):
    gain = np.power(10, -3 / 20)

    H0 = 1
    max_order = 1000

    exp_terms = np.exp(-1j * cutoff * np.arange(max_order))
    h_gains = []

    for N in range(1, max_order):
        a = H0 / N

        current_gain = np.sum(exp_terms[:N])
        h_gains.append(np.abs(a * current_gain))
        if np.abs(a * current_gain) <= gain:
            print(f"Lowpass Filter Order: {N}")
            return N


# Appliquer le filtre FIR sur le signal redressé (rectifié) pour obtenir l'enveloppe temporelle
def get_envelope(N, signal):
    fir_coeff = [1 / N for n in range(N)]  # Coefficients du filtre FIR (coefficients égaux)

    rectified_signal = np.abs(signal)  # Redressement du signal (prend la valeur absolue)

    # Appliquer la convolution pour filtrer le signal et obtenir l'enveloppe
    return np.convolve(fir_coeff, rectified_signal)


# Appliquer l'enveloppe au signal en ajustant les longueurs si nécessaire
def apply_envelope_to_signal(signal, envelope):
    if len(signal) != len(envelope):
        min_len = min(len(signal), len(envelope))
        signal = signal[:min_len]
        envelope = envelope[:min_len]
    # Appliquer l'enveloppe point par point
    return signal * envelope

    # Sauvegarder le signal dans un fichier .wav


def save_signal_to_wav(signal, sample_rate, filename="output.wav"):
    # Normalisation du signal entre -1 et 1
    signal_normalized = signal / np.max(np.abs(signal))

    # Conversion du signal en format entier 16 bits pour l'enregistrement
    signal_int16 = np.int16(signal_normalized * 32767)

    # Sauvegarde dans un fichier wav
    write(filename, sample_rate, signal_int16)


# Générer les fréquences des notes à partir de la fréquence de La#
def generate_note_frequencies(ladiese_freq):
    la_freq = ladiese_freq / 1.06  # La basse d'un demi-ton
    # Dictionnaire des fréquences des notes
    frequencies = {"do": la_freq * 0.595,
                   "do#": la_freq * 0.630,
                   "ré": la_freq * 0.667,
                   "ré#": la_freq * 0.707,
                   "mi": la_freq * 0.749,
                   "fa": la_freq * 0.794,
                   "fa#": la_freq * 0.841,
                   "sol": la_freq * 0.891,
                   "sol#": la_freq * 0.944,
                   "la": la_freq,
                   "la#": ladiese_freq,
                   "si": la_freq * 1.123}

    return frequencies


# Reproduire les premières notes de la 5e symphonie de Beethoven
def beethoven(amplitudes, phases, sample_rate, envelope, note_freqs):

    # Générer les différentes notes
    sol_audio = apply_envelope_to_signal(synthetize_note(note_freqs["sol"], amplitudes, phases, 0.4, sample_rate), envelope)
    mib_audio = apply_envelope_to_signal(synthetize_note(note_freqs["ré#"], amplitudes, phases, 1.5, sample_rate), envelope)
    fa_audio = apply_envelope_to_signal(synthetize_note(note_freqs["fa"], amplitudes, phases, 0.4,  sample_rate), envelope)
    re_audio = apply_envelope_to_signal(synthetize_note(note_freqs["ré"], amplitudes, phases, 1.5, sample_rate), envelope)
    silence_1 = create_silence(sample_rate, 0.2)

    # Construire la séquence de notes avec silences
    beethoven_audio = np.concatenate([
        sol_audio,
        sol_audio,
        sol_audio,
        mib_audio, silence_1,
        fa_audio,
        fa_audio,
        fa_audio,
        re_audio
    ])

    save_signal_to_wav(beethoven_audio, sample_rate, "beethoven.wav")


def create_silence(sampleRate, duration_s=1):
    return [0 for t in np.linspace(0, duration_s, int(sampleRate * duration_s))]


def plot_spectrum(signal, sample_rate, title="Spectrum (en dB)", harmonics=None):
    # Appliquer la FFT pour obtenir le spectre de fréquence
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_signal), 1 / sample_rate)

    # Limiter aux fréquences positives
    positive_freqs = frequencies[:len(frequencies) // 2]
    positive_amplitudes = np.abs(fft_signal)[:len(frequencies) // 2]

    # Convertir les amplitudes en dB (éviter log(0) en ajoutant un petit epsilon)
    positive_amplitudes_db = 20 * np.log10(positive_amplitudes + 1e-10)

    # Tracer le spectre en dB
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_amplitudes_db, label="Spectre en dB")
    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")

    # Ajuster les limites de l'axe y pour mieux voir les détails
    plt.ylim(np.min(positive_amplitudes_db) * 1.1, np.max(positive_amplitudes_db) * 1.1)

    # Limiter l'axe des X à 4000 Hz
    plt.xlim(0, 4000)

    # Identifier les 32 harmoniques avec de petits marqueurs sur l'axe x
    if harmonics is not None:
        for harmonic in harmonics:
            # Ajouter de petits ticks rouges en bas du graphique (petites lignes verticales seulement sur l'axe des x)
            plt.axvline(x=harmonic, color='r', linestyle='--', ymax=0.03, label=f'Harmonique: {harmonic:.2f} Hz')

    # Ne pas afficher de légende si les harmoniques sont nombreuses (limite la légende)
    if harmonics is not None and len(harmonics) <= 5:
        plt.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    # Code principal pour traiter un fichier et produire les résultats
    cutoff = np.pi / 1000
    sample_rate, signal = read_file('note_guitare_lad.wav')  # Lire le fichier audio
    duration = len(signal) / sample_rate  # calculer la duree du signal
    windowed_signal = hamming(signal)  # appliquer la fenetre de hamming

    # calcul de l enveloppe via un filtre fir
    N = get_fir_N(cutoff)
    envelope = get_envelope(N, signal)

    # Extraire les parametre sinusoidaux
    sinus_freqs, sinus_amp, sinus_phases, fundamental = get_frequencies(windowed_signal, sample_rate, 32)

    # Analyser le signal d'origine (avant synthèse)
    plot_spectrum(signal, sample_rate, title="Spectre du son analysé", harmonics=sinus_freqs[:32])

    # reproduire le signal a partir des sinusoides
    reproduced_signal = reproduce_signal(sinus_freqs, sinus_amp, sinus_phases, duration, sample_rate)

    # Analyser le signal synthétisé
    plot_spectrum(reproduced_signal, sample_rate, title="Spectre du son synthétisé", harmonics=sinus_freqs[:32])

    # Appliquer l'enveloppe et sauvegarder le signal final
    final_signal = apply_envelope_to_signal(reproduced_signal, envelope)
    save_signal_to_wav(final_signal, sample_rate)

    # Générer les fréquences des notes pour Beethoven
    note_freqs = generate_note_frequencies(fundamental)

    # Reproduire la séquence de Beethoven et la sauvegarder
    beethoven(sinus_amp, sinus_phases, sample_rate, envelope, note_freqs)