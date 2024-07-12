# import dependencies
import os 
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

# reads all audio file in the directory and store as list of dictionary
def readscpfile(filename):
	# reads a text file on the filename location
	audiofiles = open(filename, 'r').readlines()
	audiodict = []
	for file in audiofiles:
		# Strip any leading/trailing whitespace and split the line by spaces
		noisyloc, cleanloc = file.strip().split(' ')
		noisyloc, cleanloc = file.strip().split(' ')
		audiodict.append({'noisy':noisyloc, 'clean':cleanloc})
		return audiodict
	
# to get the noisy and clean audio data as a 2-channel array. 
def getaudio(audioinfo, chunk_size):
	#  Reads the audio file from the provided path, return tupple of (noisy_data, samplerate)
	noisy = sf.read(audioinfo['noisy'])[0]
	clean = sf.read(audioinfo['clean'])[0]

	# Gets the number of samples (length) in clean, and noise audio
	nsamples, csamples = noisy.shape[-1], clean.shape[-1]

	# truncate the noisy, clean audio to the same length as the smallest audio
	if nsamples >= csamples :
		noisy = noisy[:csamples]
	if csamples > nsamples :
		clean = clean[:nsamples]

	samples = len(clean)
	#to ensure samples at least the length of the chunk size.
	if samples < chunk_size:
		r = chunk_size//samples + 1
		#  Repeat the noisy audio r times. 18,000 samples (9,000 * 2). if r = 2
		noisy = np.concatenate( [noisy] * r )
		clean = np.concatenate( [clean] * r)
	
	#  Stack the noisy and clean audio vertically to create a 2-channel array.
	# audio is a 2x18,000 array 
	audio = np.vstack([noisy, clean])
	return audio

# make chunks from a audio and returns all the chunks as list 
def makechunks(audio, chunk_size, hopsize):
	chunks = [ ]
	#  If audio is a 2x18,000 array, nchannels is 2 and nsamples is 18,000.
	nchannels, nsamples = audio.shape
	if nsamples < chunk_size :
		p = chunk_size - nsamples
		# Determine the padding width for each dimension of the array.
		# p = 200, ((0,0), (0,p)) means no padding for first channels and padding 200 zeros to the end of the 2nd channel
		pad_width = ((0,0), (0,p)) if nchannels >= 2 else (0,p)
		# Pad the audio array with a constant value.
		chunk = np.pad(audio, pad_width, "constant", constant_values = 1.0e-8)
		# Get the last channel (clean audio) from the chunk.
		ref = chunk[-1, : ]
		#  Check if the silent region in the clean signal is less than 75% of the chunk size.
		if len(ref[abs(ref - 0.0) < 1.0e-6]) / (len(ref) + 1.0e-6) < 0.75:
			chunks.append(chunk)

	else : 
		s = 0
		while True:
			if s + chunk_size > nsamples :
				break
			chunk = audio[:, s:s + chunk_size]
			ref = chunk[-1 , : ]
			# consider a chunk only if the silence region in the clean signal is less than 10% of the chunksize
			if len(ref[abs(ref - 0.0)  < 1.0e-6])/ (len(ref) + 1.0e-6) < 0.75 :
				chunks.append(chunk)
			s += hopsize

		return chunks

#  splits audio files into smaller chunks and saves them to a specified directory.
# hopsize: The step size to move for each chunk (default is 8,000).
def savechunks(scpfile, destdir, chunk_size = 64000, hopsize = 8000):
	Path(destdir).mkdir(parents = True, exist_ok = True)
	# convert all audio to  list of dictionary 
	audiofiles = readscpfile(scpfile)
	chunk_count = 0
	# Iterates over each pair of audio files (noisy and clean) in audiofiles. 
	# Uses tqdm to show a progress bar with the description "Splitting audio".
	for audiopair in tqdm(audiofiles, desc = 'Splitting audio'):
		# to get the noisy and clean audio data as a 2-channel array. 
		audio = getaudio(audiopair, chunk_size)
		# make chunks from a audio and returns all the chunks as list 
		chunks = makechunks(audio, chunk_size, hopsize)
		for chunk in chunks :
			sf.write(destdir + '/audio_' + str(chunk_count)+'.wav', chunk.transpose(), 16000)
			chunk_count = chunk_count + 1
	return 
	

if __name__ == '__main__':
	print('Data preparation for splitting audio files into 4 sec chunks')
	# Create directories for training and development samples
	Path(os.path.dirname(os.getcwd() + '/Training_Samples/Train')).mkdir(parents=True, exist_ok=True)
	Path(os.path.dirname(os.getcwd() + '/Training_Samples/Dev')).mkdir(parents=True, exist_ok=True)

	# Save chunks for training data
	savechunks(os.getcwd() + '/scpfiles/Train.txt', os.getcwd() + '/Training_Samples/Train')
	# Save chunks for development data
	savechunks(os.getcwd() + '/scpfiles/Dev.txt', os.getcwd() + '/Training_Samples/Dev')
