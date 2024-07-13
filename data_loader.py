import os
import numpy as np
import soundfile as sf
from glob import glob
from librosa import stft
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
epsilon = np.finfo(np.float32).eps

datadir = os.getcwd() + '/Database/Build/Train'

class load_dataset(Dataset):
	def __init__(self, datadir ):
		#  to find all .wav files in the specified directory (datadir) 
		# and then sorts the list of file paths. return a list 
		self.audio_files = sorted(glob(datadir + '/*.wav'))

	def __len__(self) :
		return len(self.audio_files)
	
	# allows indexing into instances of the class (i.e., instance[idx])
	def __getitem__(self, idx):
		# Print the file path of the audio file at the given index
		print(self.audio_files[idx])
		
		# Read the audio file using the soundfile library
		# self.audio_files is [['path/to/audio_0.wav'], ['path/to/audio_1.wav']]
		# returns 2D NumPy array representing the audio data with shape (3, 2) (3 samples, 2 channels).
		# audio = (np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), 16000)
		audio = sf.read(self.audio_files[idx][0])
		
		# Separate the audio into noisy and clean channels
		# Assume audio is a 2D NumPy array where:
		noisy, clean = audio[:, 0].transpose(), audio[:, 1]
		
		# Convert the noisy and clean channels to PyTorch tensors
		# .squeeze() removes any singleton dimensions, but in this case, there are none to remove
		noisy, clean = torch.FloatTensor(noisy).squeeze(), torch.FloatTensor(clean).squeeze()
		
		# Create a dictionary with keys 'noisy' and 'clean' containing the corresponding tensors
		batch = {'noisy': noisy, 'clean': clean}
		
		# Return the dictionary
		return batch

	
	def norm(x):
		return x / (np.max(np.abs(x) + 1e-10 ))

	def db(x):
		xdb = 20 * np.log10(np.abs(x) + np.spacing(1))
		xdb[xdb < -120] = -120
		xdb = (xdb + 60) / 60
		return xdb

	# Returns Log Power Spectrum 
	def lps(audio):
		# 1. STFT: Converts the time-domain signal into a time-frequency representation, 
		#  producing a 2D array (spectrogram) where one axis represents time and the other represents frequency.
		# 2. Normalization: Adjusts the amplitude of the spectrogram values, often to a common scale.
		spec = norm(stft(audio, n_fft=512, hop_length=128, win_length=512, window=np.sqrt(np.hanning(512))))
		spec = db(spec)
		return spec
	

if __name__ == '__main__' :
	# Test Dataloader 
	outdir = os.getcwd() + '/Database/output/'
	Path(outdir).mkdir(parents=True, exist_ok=True)
	dataset = load_dataset(datadir)

	print(outdir)

	data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
	#  The enumerate function adds a counter to an iterable. Here, it generates pairs of (index, batch) 
	# for each batch of data loaded by data_loader.
	for i, batch in enumerate(data_loader):
		if i == 10 :
			print(batch['noisy'].shape, batch['clean'].shape)

			# Convert 'noisy' and 'clean' tensors to NumPy arrays and compute their LPS
			nspec = lps(batch['noisey'][0].cpu().numpy())
			cspec = lps(batch['clean'][0].cpu().numpy())

			# Save the LPS spectrogram images using the 'jet' colormap
			# np.flipud(nspec): Flips the spectrogram array nspec upside down,
			# with low frequencies at the bottom and high frequencies at the top. 
			# cmap='jet': Specifies the colormap to be used.
			mpimg.imsave(outdir + 'noisy_mag.png', np.flipud(nspec), cmap = 'jet' )
			mpimg.imsave(outdir + 'clean_mag.png', np.flipud(cspec), cmap='jet')
			break
