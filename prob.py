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
    return sinus_freqs, sinus_amplitudes, sinus_phases, fundamental


def reproduce_signal(frequencies, amplitudes, phases, duration, sample_rate):
    t = np.linespace(0,duration, int(sample_rate*duration))

    reproduced_signal = np.zeros_like(t)

    for i in range(len(frequencies)):
        reproduced_signal += amplitudes[i] * np.sin(2*np.pi*frequencies[i]*t + phases[i])

    return reproduced_signal


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


cutoff = np.pi/1000
sample_rate, signal = read_file('note_guitare_lad.wav')
duration = len(signal)/sample_rate
windowed_signal = hamming(signal)
N = get_fir_N(cutoff)
sinus_freqs, sinus_amp, sinus_phases, fundamental = get_frequencies(windowed_signal, sample_rate, 32)
