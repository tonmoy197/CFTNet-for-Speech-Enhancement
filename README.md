# CFTNet-for-Speech-Enhancement
This repository is involving the implementation of CFTNet architecture. The model is Trainned and tested with the audio data.

# Important Terms 
### What is sound pressure level (SPL)?
Sound pressure level (SPL) is the pressure level of a sound, measured in decibels (dB).
 It is equal to 20Log10(Root Mean Square (RMS) of sound pressure)/(reference of sound pressure)  (the reference sound pressure in air is 2 x 10-5 N/m2, or 0,00002 Pa). 

### What is STFT(Short Time Fourier Transformation) and Normalization ?
STFT: Converts the time-domain signal into a time-frequency representation, producing a 2D array (spectrogram) where one axis represents time and the other represents frequency.

Normalization: Adjusts the amplitude of the spectrogram values, often to a common scale.

### What is Log Power Spectrum (LPS) ?
The Log Power Spectrum is a representation of the signal's power across different frequencies, expressed in decibels (dB). It is commonly used in audio signal processing for tasks like speech recognition, noise reduction, and audio analysis.