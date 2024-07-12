import librosa as lr 
import numpy as np
import soundfile as sf 
import os
from pathlib import Path

#calculates the root mean square (RMS) energy of an audio signal x and converts it to decibels (dB)
def rms_energy(x):
	return 10*np.log10((1e-12 + x.dot(x))/ (len(x)))

# adjusts the sound pressure level (SPL) of an audio signal to a desired level
def SPL_cal(x, SPL) :
	# x: The audio signal (an array of samples).
	# SPL: The desired sound pressure level in decibels (dB).
	# 20Log10(Root Mean Square (RMS) of sound pressure)/(reference of sound pressure)
	SPL_before = 20 * np.log10(np.sqrt(np.mean(x**2)) / (20 * 1e-6)) # in dB
	# Scales the original signal x by the linear gain factor to achieve the desired SPL.
	y = x * 10 ** ((SPL - SPL_before) / 20)
	return y

def add_noise(signal, noise, fs, snr, signal_energy = 'rms'):
	# Select random section of noise
	if len(noise) != len(signal) :
		# Generates a random integer index idx within the range [0, len(noise) - len(signal))
		idx = np.random.randint(0, len(noise) - len(signal)) # negative value exceptable
		noise = noise[idx : idx + len(signal)]

	# find rms energy of both noise and signal
	N_dB = rms_energy(noise)
	if signal_energy =='rms':
		S_dB = rms_energy(signal)
	else :
		raise ValueError('Signal_energy has to be either "rms" or "P.56"')
	
	# target RMS energy of the noise signal that, when added to the signal
	N_new = S_dB - snr
	# convert dB values back to linear scale (amplitude).
	noise_scaled = 10**(N_new/20) * noise/10 **(N_dB/20)
	noisy_signal = signal + noise_scaled
	return noisy_signal


def datagenerator(in_path, out_path, noise_file, snr, num_audio_sample, sample_margin, fs):
	for idx in range(num_audio_sample) :
		# data for input
		out_file_noisy = "Noisy_" + noise_file + "_" + str(snr)+"_dB_"+str(idx + sample_margin)
		out_file_clean = "Clean_" + str(idx + sample_margin)
		clean, fs = lr.load(in_path + '/Clean/Clean_'+str(idx+sample_margin)+'.wav', sr = fs)
		noise, fs = lr.load(in_path + '/Different_Noise/' + noise_file + '_noise.wav', sr = fs)

		clean = SPL_cal(clean, 65)
		noisy = add_noise(clean, noise, fs, snr)
		noisy = SPL_cal(noisy, 65)

		sf.write(out_path + '/Noisy/' + out_file_noisy + ".wav", noisy, fs)
		sf.write(out_path + '/Clean/' + out_file_clean + ".wav", clean, fs)

		print(str(idx))


		