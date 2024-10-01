from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks
from scipy.io.wavfile import write


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


def reproduce_signal(fundamental, amplitudes, phases, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate*duration))

    reproduced_signal = np.zeros_like(t)

    for i in range(1, len(amplitudes)):
        reproduced_signal += amplitudes[i-1] * np.sin(2*np.pi*i*fundamental*t + phases[i-1])

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


def get_envelope(N, signal):
    fir_coeff = [1/N for n in range(N)]

    rectified_signal = np.abs(signal)

    return np.convolve(fir_coeff, rectified_signal)


def apply_envelope_to_signal(signal, envelope):
    if len(signal) != len(envelope):
        min_len = min(len(signal), len(envelope))
        signal = signal[:min_len]
        envelope = envelope[:min_len]

    return signal * envelope


def save_signal_to_wav(signal, sample_rate, filename="output.wav"):
    # Normalisation du signal entre -1 et 1
    signal_normalized = signal / np.max(np.abs(signal))

    # Conversion du signal en format entier 16 bits pour l'enregistrement
    signal_int16 = np.int16(signal_normalized * 32767)

    # Sauvegarde dans un fichier wav
    write(filename, sample_rate, signal_int16)


def generate_note_frequencies(ladiese_freq):
    la_freq = ladiese_freq / 1.06

    frequencies = { "do"   : la_freq * 0.595,
                    "do#"  : la_freq * 0.630,
                    "ré"   : la_freq * 0.667,
                    "ré#"  : la_freq * 0.707,
                    "mi"   : la_freq * 0.749,
                    "fa"   : la_freq * 0.794,
                    "fa#"  : la_freq * 0.841,
                    "sol"  : la_freq * 0.891,
                    "sol#" : la_freq * 0.944,
                    "la"   : la_freq,
                    "la#"  : ladiese_freq,
                    "si"   : la_freq * 1.123 }

    return frequencies


def beethoven(amplitudes, phases, sample_rate, envelope, note_freqs):
    sol_audio = apply_envelope_to_signal(reproduce_signal(note_freqs["sol"], amplitudes, phases, 0.4, sample_rate), envelope)
    mib_audio = apply_envelope_to_signal(reproduce_signal(note_freqs["ré#"], amplitudes, phases, 1.5, sample_rate), envelope)
    fa_audio = apply_envelope_to_signal(reproduce_signal(note_freqs["fa"], amplitudes, phases, 0.4,  sample_rate), envelope)
    re_audio = apply_envelope_to_signal(reproduce_signal(note_freqs["ré"], amplitudes, phases, 1.5, sample_rate), envelope)
    silence_1 = create_silence(sample_rate, 0.2)

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

cutoff = np.pi/1000
sample_rate, signal = read_file('note_guitare_lad.wav')
duration = len(signal)/sample_rate
windowed_signal = hamming(signal)
N = get_fir_N(cutoff)
envelope = get_envelope(N, signal)
sinus_amp, sinus_phases, fundamental = get_frequencies(windowed_signal, sample_rate, 32)
reproduced_signal = reproduce_signal(fundamental, sinus_amp, sinus_phases, duration, sample_rate)

final_signal = apply_envelope_to_signal(reproduced_signal, envelope)
save_signal_to_wav(final_signal, sample_rate)
note_freqs = generate_note_frequencies(fundamental)
beethoven(sinus_amp, sinus_phases, sample_rate, envelope, note_freqs)