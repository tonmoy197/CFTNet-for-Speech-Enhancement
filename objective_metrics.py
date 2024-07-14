import numpy as np
import torch
from torch import nn
# from torch import mean, norm
from torch.nn.functional import unfold , pad

import torchaudio
from pystoi.stoi import FS, N_FRAME, NUMBAND, MINFREQ, N, BETA, DYN_RANGE
from pystoi.utils import thirdoct
EPS = 1e-8

# calculate log spectral distance

def lsd(x, y):
	fft_1 = np.fft.fft(x)
	fft_2 = np.fft.fft(y)

	# compute power spectra 
	power_spectrum_1 = np.abs(fft_1) ** 2
	power_spectrum_2 = np.abs(fft_2) ** 2

	# Compute LSD : sqrt(mean( (1/2*pi) * (10*log(power1)/power2)) ^ 2)
	log_spectral_distance = np.mean(
        np.sqrt((1 / (2 * np.pi)) * (np.power(10 * np.log10(power_spectrum_1 / power_spectrum_2), 2)))
		)

	return log_spectral_distance

# calculate Negated Short Term Objective Intelligibility (STOI) metric
# def NegSTOI(nn.Module ):
# 	pass


# meanvar_norm = x / sqrt(sum((x-mean)^2))
def meanvar_norm(x, mask = None, dim = -1) :
	# calculates the mean of x along the specified dimension dim, considering the mask if provided.
	# Subtracting the mean centers the data around zero.
	x = x - masked_mean(x, dim = dim , keepdim = True)
	# Dividing by the Norm L2(Standard Deviation)

	x = x / (masked_norm(x, p = 2, dim = dim, keepdim = True, mask = mask) + EPS)
	return x 

# calculates the mean of a tensor x along a specified dimension dim, considering a mask
def masked_mean(x, dim = -1, mask = None, keepdim = False):
	if mask is None :
		return torch.mean(x, dim=dim, keepdim =keepdim)
	return torch.sum(x * mask, axis = dim, keepdim = keepdim) / (mask.sum(dim = dim, keepdim = keepdim) + EPS)

# calculates the norm of a tensor x along a specified dimension dim, considering a mask
def masked_norm(x, p =2, dim = -1, mask = None, keepdim = False):
	if mask is None : 
		return torch.norm(x, p = p, dim = dim, keepdim= keepdim)
	# L2 Normalization : norm = x / sqrt(sum((x)^2))
	return torch.norm(x * mask, p = p, dim = dim, keepdim=keepdim)

# Scale-Invariant Signal-to-Noise Ratio (SI-SNR) between two sources:
#  a reference source (source) and an estimated source (estimate_source). 
def si_snr(source, estimate_source, eps=1e-5):
    # Ensure inputs are 2-dimensional (Batch size B x Length T)
    source = source.squeeze(1)
    estimate_source = estimate_source.squeeze(1)

    B, T = source.size()  # B: Batch size, T: Length of each source

    # Calculate the energy of the reference source
    source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B x 1
    
    # Dot product between estimated source and reference source
    dot = torch.matmul(estimate_source, source.t())  # B x B
    
    # Scale the reference source using the dot product and its energy
    s_target = torch.matmul(dot, source) / (source_energy + eps)  # B x T
    
    # Calculate the estimation error (noise) between estimated source and reference source
    e_noise = estimate_source - source  # B x T
    
    # Calculate SI-SNR in decibels
    snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)
    
    # Loss function: negative mean SI-SNR (to minimize)
    loss = 0 - torch.mean(snr)
    
    return loss

class SiSNR(object):
	# allows an instance of SiSNR is called with source and estimate_source as arguments, it invokes si_snr 
	def __call__(self, source, estimate_source):
		return si_snr(source, estimate_source)

# Negated Short Term Objective Intelligibility (STOI) metric
# Computes the negative STOI loss between estimated target signals (est_targets) and clean target signals (targets).
# This implementation deviates from the original STOI metric for computational efficiency
# Uses VAD for handling silent frames efficiently within batches.
class NegSTOILoss(nn.Module) : 
	def __init__(self, 
			  sample_rate:int,             # sample rate of audio input
			  use_vad : bool = True,       # Whether to use simple VAD (see Notes)
			  extended : bool = False,     # Whether to compute extended version
			  do_resample : bool = True):  # Whether to resample audio input to `FS`
		super().__init__()

		# Independent form FS
		self.sample_rate = sample_rate
		self.use_vad = use_vad
		self.extended = extended
		self.do_resample = do_resample
		self.intel_frames = N
		self.beta = BETA
		self.dyn_range = DYN_RANGE

		# Dependent from FS
		if self.do_resample:
			sample_rate = FS
			# Change sample rate from a sampling_rate to FS using sinc interpolation method.
			self.resample = torchaudio.transforms.Resample(
				orig_freq=self.sample_rate,
				new_freq = FS,
				resampling_method =  "sinc_interp_hann"
			)

			# // --> used to floor division
			self.win_len = (N_FRAME * sample_rate) // FS
			self.nfft = 2 * self.win_len

			# defining Hanning window in numpy and convert to tensor of float type
			win = torch.from_numpy(np.hanning(self.win_len + 2)[1:-1]).float()
			# Converts the win tensor into a nn.Parameter instance, making it a learnable parameter 
			# requires_grad=False ensures that this parameter does not have gradients computed during backpropagation, 
			# indicating it is a fixed parameter.
			self.win = nn.Parameter(win, requires_grad=False)

			obm_mat = thirdoct(sample_rate, self.nfft, NUMBAND, MINFREQ)[0]
			self.obm = nn.Parameter(torch.from_numpy(obm_mat).float(), requires_grad=False)

	#  the method can handle inputs with more than 2 dimensions while internally working with simpler 2D shapes.
	def forward(self, est_targets : torch.Tensor, targets : torch.Tensor ) -> torch.Tensor:
		# est_targets , targets : (batch_size, wav_len)
		# checking for size mismatch
		if targets.shape != est_targets.shape :
			if targets.shape[0] == 1 :
				targets = targets.unsqueeze(0)
			else : 
				raise RuntimeError('Targets and est_targets should have the same shape,found {} and '
						'{}'.format(targets.shape, est_targets.shape))
		
		# Computer STOI Loss without batch size 
		# it reshapes them to 2D by adding an extra dimension and calls forward again.
		if targets.ndim == 1 :
			return self.forward(est_targets[None], targets[None])[0]

		# Pack additional dimensions in batch and unpack after forward 
		if targets.ndim > 2 :
			# inner captures all dimensions except the last one
			*inner, wav_len = targets.shape
			return self.forward(
				est_targets.view(-1, wav_len),
				targets.view(-1, wav_len)
			).view(inner)
		
		# 1. updating targets, est_targets when resample needed
		if self.do_resample and self.sample_rate != FS: 
			targets = self.resample(targets)
			est_targets = self.resample(est_targets)

		# 2. Taking STFT 
		x_spec = self.stft(targets, self.win, self.nfft)
		y_spec = self.stft(est_targets, self.win, self.nfft)

		# 3. Apply OB matrix to the spectrograms as in EQ (i)
		x_tob = torch.matmul(self.obm, torch.norm(x_spec, 2, -1) ** 2).pow(0.5)
		y_tob = torch.matmul(self.obm, torch.norm(y_spec, 2, -1) ** 2).pow(0.5)

		# 4. Perform N-Frame Segmentation --> (batch, 15, N, n_chunks)
		batch = targets.shape[0]
		# creates sliding windows (or patches) from the input tensor.
		x_seg = unfold(
			# If x_tob has a shape of (batch, time)--> (batch, time, 1)
			x_tob.unsqueeze(2),
			# the size of the sliding window.
			kernel_size=(1, self.intel_frames),
			# stride=(1,1) specifies the step size of the sliding window.
			stride=(1,1)
		).view(batch, x_tob.shape[1], N, -1)

		y_seg = unfold(
			y_tob.unsqueeze(2),
			kernel_size=(1, self.intel_frames),
			stride=(1,1)
		).view(batch, y_tob.shape[1], N, -1)

		# 5. Voice activity Detection
		if self.use_vad :
			# detech silent frames (boolean mask of shape(batch, 1, frame_idx))
			mask = self.detect_silent_frames(targets, self.dyn_range, self.win_len, self.win_len//2)
			mask = pad(mask, [0, x_tob.shape[-1] - mask.shape[-1]])
			# Unfold on the mask, to float and mean per frame 
			mask_f = unfold(
				mask.unsqueeze(2).float(),
				kernel_size=(1, self.intel_frames),
				stride=(1,1).view(batch, 1, N, -1)
			)

		else :
			mask_f = None

		if self.extended:
			# 6. Normalizaion and Clipping
			# Normalize rows and colums of intermediate intelligiblity frames 
			x_n = self.rowcol_norm(x_seg, mask = mask_f)
			y_n = self.rowcol_norm(y_seg, mask = mask_f)
			# 7. corelation Computation 
			corr_comp = x_n * y_n
			correlation = self.intel_frames * x_n.shape[-1]
		else :
			# 6. Normalizaion and Clipping
			norm_const = (masked_norm(x_seg, dim = 2, keepdim=True, mask=mask_f)/
				 masked_norm(y_seg, dim = 2, keepdim=True, mask = mask_f) + EPS)
			y_seg_normed = y_seg * norm_const
			# clip as described in [1]
			clip_val = 10 ** (-self.beta/20)
			y_prim = torch.min(y_seg_normed, x_seg * (1 + clip_val))
			# Mean / Var normalize vectors 
			y_prim = meanvar_norm(y_prim, dim = 2, mask = mask_f)
			x_seg = meanvar_norm(x_seg, dim = 2, mask = mask_f)

			# 7. corelation Computation 
			corr_comp = y_prim * x_seg
			correlation = x_seg.shape[1] * x_seg.shape[-1]

		# 8. Computing NegSTOILoss w. or w/o VAD
		sum_over = list(range(1, x_seg.ndim)) # Keep batch dim
		if self.use_vad :
			corr_comp = corr_comp * mask_f
			correlation = correlation * mask_f.mean() + EPS
		
		return -torch.sum(corr_comp, dim=sum_over) / correlation
	
	@staticmethod
	def detect_silent_frames(x, dyn_range, frame_len, hop) :
		'''Detects silent frames of i/p tensor. and excludes 
		if its energy is lower than max(energy)-dyn_range 
		Args :
			x(torch.Tensor) : Batch of original speech wav file (Batch, Time)
			dyn_range : Energy range to determine which frame is silent
			frame_len : Window size for energy evalution
			hop : Hop size for energy evaluation 
		Returns : 
			torch.BoolTensor, framewise_mask
		'''
		x_frames = unfold(
			x[:, None, None, :], 
			kernel_size=(1, frame_len),
			stride=(1, hop),
		)[..., :-1]

		# computes energies in dB
		x_energies = 20 * torch.log10(torch.norm(x_frames, dim=1, keepdim=True)+ EPS)
		# Find boolean mask of energies lower than dynamic_range dB
		# With respect to maximum clean speech energy frame 
		mask = (torch.max(x_energies, dim = 2 , keepdim=True)[0] - dyn_range-x_energies) < 0
		return mask 

	def stft(x, win, fft_size, overlap=4):
		win_len = win.shape[0]
		hop = int(win_len / overlap)
		# Last frame not taken because NFFT size is larger, torch bug IMO.
		x_padded = torch.nn.functional.pad(x, pad=[0, hop])
		return torch.stft(x_padded, fft_size, hop_length=hop, window=win,
							center=False, win_length=win_len)

	@staticmethod
	def rowcol_norm(x, mask = None) :
		"""Mean / Variance normalize axis 2 and 1 of input vector """
		for dim in [2, 1] :
			x = meanvar_norm(x, mask=mask, dim=dim)
		
		return x


	
if __name__ == '__main__':
	x = torch.tensor(np.random.randn(1, 20000))
	y = torch.tensor(np.random.randn(1, 20000))
	lsd_score = lsd(x, y)
	meanvar = meanvar_norm(x)
	si_snr_val = si_snr(x, y)
	print(lsd_score)
	print(meanvar)
	print(si_snr_val)
	

	# Define example parameters for negSTOILoss
	sample_rate = 16000
	FS = 8000
	N_FRAME = 400
	NUMBAND = 24
	MINFREQ = 20
	N = 15
	BETA = 15
	DYN_RANGE = 40
	EPS = 1e-8

	# Create example tensors for est_targets and targets
	batch_size = 2
	wav_len = 1000

	est_targets = torch.randn(batch_size, wav_len)  # Example estimated target signals
	targets = torch.randn(batch_size, wav_len)      # Example clean target signals

	# Instantiate NegSTOILoss module
	neg_stoi_loss = NegSTOILoss(
		sample_rate=sample_rate,
		use_vad=True,
		extended=False,
		do_resample=True
	)

	# Compute the loss
	loss = neg_stoi_loss(est_targets, targets)
	print("NegSTOILoss:", loss.item())
