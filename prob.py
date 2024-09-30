from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks


def read_file(filename):
    sample_rate, signal = wavfile.read(filename)
    return sample_rate, signal


def hamming(signal):
    hamming_window = np.hamming(len(signal))
    windowed_signal = np.multiply(signal, hamming_window)
    return windowed_signal


def get_frequencies(signal, sample_rate, sinus_count):
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_signal), 1/sample_rate)

    index_fundamental = np.argmax(np.abs(fft_signal))
    fundamental = np.abs(frequencies[index_fundamental])
    print("La# fundamental frequency: " + str(fundamental))


    # Limiter les harmoniques trouvés à harmonicsCount
    sinus_freqs = []
    sinus_amplitudes = []
    sinus_phases = []

    # The first frequency is the fundamental
    sinus_freqs.append(fundamental)
    sinus_amplitudes.append(np.abs(fft_signal[index_fundamental]))
    sinus_phases.append(np.angle(fft_signal[index_fundamental]))

    # Now find the harmonics (multiples of the fundamental frequency)
    for i in range(2, sinus_count + 1):  # Start from the 2nd harmonic
        harmonic_freq = fundamental * i
        closest_idx = np.argmin(np.abs(frequencies - harmonic_freq))  # Find the closest frequency bin

        sinus_freqs.append(frequencies[closest_idx])
        sinus_amplitudes.append(np.abs(fft_signal[closest_idx]))
        sinus_phases.append(np.angle(fft_signal[closest_idx]))

    print("Frequencies:", sinus_freqs)
    print("Amplitudes:", sinus_amplitudes)
    print("Phases:", sinus_phases)
    return sinus_amplitudes, sinus_phases, fundamental


sample_rate, signal = read_file('note_guitare_lad.wav')
windowed_signal = hamming(signal)
sinus_amp, sinus_phases, fundamental = get_frequencies(windowed_signal, sample_rate, 32)