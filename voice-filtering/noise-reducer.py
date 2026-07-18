import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("noisereduce not installed. Noise reduction will be skipped.")

def butter_bandpass(lowcut=80.0, highcut=4000.0, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=80.0, highcut=4000.0, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def reduce_noise(audio_array, sr, noise_clip=None, use_noisereduce=True, prop_decrease=0.35):
    """
    Reduce noise from audio using noisereduce if available.
    If noise_clip is provided, it is used as the noise profile.
    """
    if use_noisereduce and NOISEREDUCE_AVAILABLE:
        if noise_clip is not None:
            return nr.reduce_noise(
                y=audio_array,
                sr=sr,
                y_noise=noise_clip,
                stationary=False,
                prop_decrease=prop_decrease,
            )
        return nr.reduce_noise(
            y=audio_array, sr=sr, stationary=False, prop_decrease=prop_decrease
        )
    # If noisereduce is unavailable or disabled, return original
    return audio_array

def process_audio(input_file, output_file, lowcut=80.0, highcut=4000.0, use_noise_reduction=True):
    """
    Read audio, apply bandpass filter and optional noise reduction, save output.
    """
    audio_array, sr = sf.read(input_file)
    
    # Bandpass filter
    filtered_audio = bandpass_filter(audio_array, lowcut, highcut, fs=sr)
    
    # Optional noise reduction
    noise_clip = filtered_audio[0:int(0.5*sr)]  # assume first 0.5s is mostly noise
    filtered_audio = reduce_noise(filtered_audio, sr, noise_clip=noise_clip, use_noisereduce=use_noise_reduction)
    
    sf.write(output_file, filtered_audio, sr)
    return output_file

if __name__ == "__main__":
    # Example usage
    process_audio("user_recording.wav", "filtered_recording.wav")
    print("Audio filtering complete!")