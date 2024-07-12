import librosa as lr 
import numpy as np
import soundfile as sf 
import os
from pathlib import Path


def rms_energy(x):
	return 10*np.log10((1e-12 + x.dot(x))/ (len(x)))

# adjusts the sound pressure level (SPL) of an audio signal to a desired level
def SPL_cal(x, SPL) :
	SPL_before = 20 * np.log10(np.sqrt(np.mean(x**2)) / (20 * 1e-6))
	y = x * 10 ** ((SPL - SPL_before) / 20)
	return y

