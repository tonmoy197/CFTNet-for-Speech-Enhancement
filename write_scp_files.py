import os
from pathlib import Path

rootdir = os.getcwd() + '/Database/Build/'
Path(os.path.dirname(os.getcwd() + '/Database/scpfiles/')).mkdir(parents=True, exist_ok=True)
txt_file_directory = os.getcwd() + '/Database/scpfiles/'

## ------------------Train -----------------##
# Store all the directories of the train data to a txt file 
snr_all = [0, 5]
noise_type_all = ['Babble', 'Car']
audio_sample = 10
# If the specified file (e.g., Train.txt) does not exist in the given directory,
#  Python will create a new file with that name.
file = open(txt_file_directory + 'Train.txt', 'w')

for noise_type in noise_type_all:
	for snr in snr_all:
		for i in range(audio_sample):
			signal_path = rootdir + '/Train/Clean/Clean_'+str(i+1)+'.wav'
			noisy_path = rootdir + '/Train/Noisy/Noisy_' + noise_type + '_' + str(snr) + '_dB_' + str(i + 1) + '.wav'
			L = noisy_path + " " + signal_path
			file.writelines(L)
			file.write('\n')

file.close()

## ------------------ Dev -----------------------##
# Store all the directories of the Dev data to a txt file 
snr_all = [0, 5]
noise_type_all = ['Babble', 'Car']
audio_sample = 5
file = open(txt_file_directory+ 'Dev.txt', 'w')

for noise_type in noise_type_all :
	for snr in snr_all:
		for i in range(audio_sample):
			signal_path = rootdir + '/Dev/Clean/Clean_'+str(i+1)+'.wav'
			noisy_path = rootdir + '/Train/Noisy/Noisy_' + noise_type + '_' + str(snr) + '_dB_' + str(i + 1) + '.wav'
			L = noisy_path + " " + signal_path
			file.writelines(L)
			file.write('\n')

file.close()

## ------------------ Test -----------------------##
# Store all the directories of the test data to a txt file 
snr_all = [0, 5]
noise_type_all = ['Babble', 'Car']
audio_sample = 5
file = open(txt_file_directory+ 'Test.txt', 'w')

for noise_type in noise_type_all :
	for snr in snr_all:
		for i in range(audio_sample):
			enhanced_path = rootdir + '/Test/Enhanced/Enhanced_' + noise_type +  '_' + str(snr) + '_dB_' + str(i + 1) + '.wav'
			signal_path = rootdir + '/Test/Clean/Clean_'+str(i+1)+'.wav'
			noisy_path = rootdir + '/Train/Noisy/Noisy_' + noise_type + '_' + str(snr) + '_dB_' + str(i + 1) + '.wav'
			L = noisy_path + " " + signal_path
			file.writelines(L)
			file.write('\n')

file.close()



