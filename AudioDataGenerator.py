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
		# data for input, 
		out_file_noisy = "Noisy_" + noise_file + "_" + str(snr)+"_dB_"+str(idx + sample_margin)
		out_file_clean = "Clean_" + str(idx + sample_margin)
		clean, fs = lr.load(in_path + '/Clean/Clean_SPK1_S'+str(idx+sample_margin)+'.wav', sr = fs)
		noise, fs = lr.load(in_path + '/Different_Noise/' + noise_file + '_noise.wav', sr = fs)

		# clean, noisy is calibrated to an SPL of 65 dB
		clean = SPL_cal(clean, 65)
		noisy = add_noise(clean, noise, fs, snr)
		noisy = SPL_cal(noisy, 65)

		# Creating paths if they not exits 
		Path(os.path.dirname(out_path + '/Clean/')).mkdir(parents=True, exist_ok= True)
		Path(os.path.dirname(out_path + '/Noisy/')).mkdir(parents=True, exist_ok=True)

		# storing the generated file 
		sf.write(out_path + '/Noisy/' + out_file_noisy + ".wav", noisy, fs)
		sf.write(out_path + '/Clean/' + out_file_clean + ".wav", clean, fs)
		print(str(idx), end=" ")

	print("\n")
	
if __name__ == '__main__' :
	path = os.getcwd()
	in_path = path + '/Database/Original_Samples'
	out_path = path + '/Database/Build'
	
	Path(os.path.dirname(out_path + '/Test/Enhanced/')).mkdir(parents=True, exist_ok=True)


# ----------------Train, Dev, Test ----------------#
fs = 16000
noise_type = ['Babble', 'Car']
path = os.getcwd()

# --------------------Train---------------------------#
print("---------Started generating Train data ---------")
snr_all = [0, 5]
num_audio_sample =  10
sample_margin = 1 
for noise_file in noise_type :
	for snr in snr_all:
		datagenerator(in_path, out_path+'/Train/', noise_file, snr, num_audio_sample, sample_margin, fs )
		

# --------------------Dev ----------------------------#
print("---------Started generating Dev data ---------")

snr_all = [0, 5]
num_audio_sample = 5
sample_margin = 11
for noise_file in noise_type:
	for snr in snr_all:
		datagenerator(in_path, out_path + '/Dev/' , noise_file, snr, num_audio_sample, sample_margin, fs)


# -------------------- Test ----------------------------#
print("---------Started generating Test data ---------")

snr_all = [5, 10]
num_audio_sample = 5
sample_margin = 16
for noise_file in noise_type:
	for snr in snr_all:
		datagenerator(in_path, out_path + '/Test/' , noise_file, snr, num_audio_sample, sample_margin, fs)

print("----------Generation Completed --------------- ")