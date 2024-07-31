from modules import *
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(9999)
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parameters()])
    return np.round(neles / 10 ** 6 if Mb else neles, 2)


# Network - 1 : >>>>>>>>>>>>>>>>>>>>>>>>>>>> CFTNet >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class CFTNet(torch.nn.Module):
	def __init__(self, L = 256, N = 256, H = 128, Mask = [5, 7], B = 24, F_dim = 129):
		super().__init__()
		self.name = 'CFTNet'
		self.f_taps = list(range(-Mask[0] // 2 + 1, Mask[0] // 2 + 1))
		self.t_taps = list(range(-Mask[1] // 2 + 1, Mask[1] // 2 + 1))

		self.stft = STFT(frame_len = L, frame_hop = H, num_fft = N)
		self.istft = iSTFT(frame_len = L, frame_hop = H, num_fft = N)

		# Complex Encoder Layer and FTB Layer 
		self.enc1 = ComplexEncoder(1, 1 * B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)

		self.FTB1 = ComplexFTB(math.ceil(F_dim/2), channels=1*B) # First FTB Layer
		self.enc2 = ComplexEncoder(1*B, 2*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.enc3 = ComplexEncoder(2*B, 2*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)

		self.FTB2 = ComplexFTB(math.ceil(F_dim/8), channels=2*B) # Second FTB Layer
		self.enc4 = ComplexEncoder(2*B, 3*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.enc5 = ComplexEncoder(3*B, 3*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)

		self.FTB3 = ComplexFTB(math.ceil(F_dim/32), channels=3*B) # Third FTB Layer
		self.enc6 = ComplexEncoder(3*B, 4*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.enc6 = ComplexEncoder(4*B, 4*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		
		self.FTB3 = ComplexFTB(math.ceil(F_dim/128), channels=3*B) # Fourth FTB Layer
		self.enc6 = ComplexEncoder(4*B, 8*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)

		# Skip Connections 
		self.skip1 = SkipConnection(8 * B, num_convblocks=4)
		self.skip2 = SkipConnection(4 * B, num_convblocks=4)
		self.skip3 = SkipConnection(4 * B, num_convblocks=4)
		self.skip4 = SkipConnection(3 * B, num_convblocks=3)
		self.skip5 = SkipConnection(3 * B, num_convblocks=3)
		self.skip6 = SkipConnection(2 * B, num_convblocks=2)
		self.skip7 = SkipConnection(2 * B, num_convblocks=2)
		self.skip8 = SkipConnection(1 * B, num_convblocks=1)

		# Decoder Connection
		self.dec1 = ComplexDecoder(16*B, 8*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True,output_padding=(1,0))
		self.dec2 = ComplexDecoder(12*B, 8*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.dec3 = ComplexDecoder(12*B, 4*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.dec4 = ComplexDecoder(7*B, 3*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.dec5 = ComplexDecoder(6*B, 3*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.dec6 = ComplexDecoder(5*B, 2*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.dec7 = ComplexDecoder(4*B, 2*B, kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)
		self.dec8 = ComplexDecoder(3*B, Mask[0] * Mask[1], kernel_size=(3,3), stride=(2,1), padding=(1,1), bias=True)


	def cat(self, x, y, dim ):
		real = torch.cat([x.real, y.real], dim)
		imag = torch([x.imag, y.imag], dim)
		return ComplexTensor(real, imag)

	def deepfiltering(self, deepfilter, cplxinput):
		deepfilter = deepfilter.permute(0, 2, 3, 1)
		real_tf_shift = torch.stack(
            [torch.roll(cplxInput.real, (i, j), dims=(1, 2)) for i in self.f_taps for j in self.t_taps], 3).transpose(-1, -2)

		imag_tf_shift = torch.stack(
			[torch.roll(cplxInput.imag, (i, j), dims=(1, 2)) for i in self.f_taps for j in self.t_taps], 3).transpose( -1, -2)

		imag_tf_shift += 1e-10

		cplx_input_shift = eninsum('bftd, bfdt -> bft', [deepfilter.conj(), cplx_input_shift])
		return est_complex

	def forward(self, audio, verbose = False):
		""" 
		batch : tensor of shape (Batch_size x Channels x Number_samples)
		"""
		if verbose : print('*' * 60)
		if verbose : print('Input Audio Shape :', audio.shape)
		if verbose : print('*' * 60)


		# Passing through Encoder and FTB Block 
		_,_,real, imag = self.stft(audio)
		cplxin = ComplexTensor(real, imag)
		enc1 = self.enc1(cplxin.unsqueeze(1))
		FTB1 = self.FTB1(enc1)
		enc2 = self.enc3(FTB1)
		FTB3 = self.FTB2(enc2)
		enc4 = self.enc4(FTB3)
		enc5 = self.enc5(enc4)
		FTB3 = self.FTB3(enc5)
		enc6 = self.enc6(FTB3)
		enc7 = self.enc6(enc6)
		FTB4 = self.FTB4(enc7)
		enc8 = self.enc8(FTB4)

		# Expanding Path 
		gru = self.GRU(enc8.squeeze(2)).unsqueeze(2)
		dec = self.dec1(self.cat(gru, self.skip1(enc8)), 1)
		dec = self.dec2(self.cat(dec, self.skip2(enc7), 1))
		dec = self.dec3(self.cat(dec, self.skip3(enc6), 1))
		dec = self.dec4(self.cat(dec, self.skip4(enc5), 1))
		dec = self.dec5(self.cat(dec, self.skip5(enc4), 1))
		dec = self.dec6(self.cat(dec, self.skip6(enc3), 1))
		dec = self.dec7(self.cat(dec, self.skip7(enc2), 1))
		dec = self.dec8(self.cat(dec, self.skip8(enc1), 1))
		

		deepfilter = ComplexTensor(dec.real, dec.imag)
		enhanced = self.deepfiltering(deepfilter, cplxin)
		enh_mag , enh_phase = enhanced.abs(), enhanced.angle()
		audio_enh = self.istft(enh_mag, enh_phase, squeeze = True)

		return audio_enh


# Network - 2 : >>>>>>>>>>>>>>>>>>>>>>>>>>>> DCCTN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class DCCTN(torch.nn.Module):
	"""
		FTBComplexSkipConvNet + Transformer 
		Complex Skip Convolution 
		It uses only two FTB Layers; one in the first layer and one on the last layer 
		Instead of using LSTM, it uses transformer 
	"""

	def __init__(self, L = 256, N = 128, Mask = [5,7], B = 24, F_dim = 129):
		super().__init__()
		self.name = 'DCCTN'

		self.f_taps = list(range(-Mask[0] // 2 + 1, Mask[0] // 2 + 1))
		self.t_taps = list(range(-Mask[1] // 2 + 1, Mask[1] // 2 + 1))

		self.enc1 = ComplexEncoder(1, 1 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB1 = ComplexFTB(math.ceil(F_dim / 2), channels=1 * B)  # First FTB layer
		self.enc2 = ComplexEncoder(1 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB2 = ComplexFTB(math.ceil(F_dim / 4), channels=2 * B)  # Second FTB layer
		self.enc3 = ComplexEncoder(2 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB3 = ComplexFTB(math.ceil(F_dim / 8), channels=2 * B)  # Third FTB layer
		self.enc4 = ComplexEncoder(2 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB4 = ComplexFTB(math.ceil(F_dim / 16), channels=3 * B)  # Fourth FTB layer
		self.enc5 = ComplexEncoder(3 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB5 = ComplexFTB(math.ceil(F_dim / 32), channels=3 * B)  # Fifth FTB layer
		self.enc6 = ComplexEncoder(3 * B, 4 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB6 = ComplexFTB(math.ceil(F_dim / 64), channels=4 * B)  # Sixth FTB layer
		self.enc7 = ComplexEncoder(4 * B, 4 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.FTB7 = ComplexFTB(math.ceil(F_dim / 128), channels=4 * B)  # Seventh FTB layer
		self.enc8 = ComplexEncoder(4 * B, 8 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.TB = ComplexTransformer(nhead=1, num_layer=2)  # d_model = x.shape[3]
		self.GRU = ComplexGRU(8 * B, 8 * B, num_layers=2)

		self.skip1 = SkipConnection(8 * B, num_convblocks=4)
		self.skip2 = SkipConnection(4 * B, num_convblocks=4)
		self.skip3 = SkipConnection(4 * B, num_convblocks=3)
		self.skip4 = SkipConnection(3 * B, num_convblocks=3)
		self.skip5 = SkipConnection(3 * B, num_convblocks=2)
		self.skip6 = SkipConnection(2 * B, num_convblocks=2)
		self.skip7 = SkipConnection(2 * B, num_convblocks=1)
		self.skip8 = SkipConnection(1 * B, num_convblocks=1)

		self.dec1 = ComplexDecoder(16 * B, 8 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True,
									output_padding=(1, 0))
		self.dec2 = ComplexDecoder(12 * B, 8 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.dec3 = ComplexDecoder(12 * B, 4 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.dec4 = ComplexDecoder(7 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.dec5 = ComplexDecoder(6 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.dec6 = ComplexDecoder(5 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.dec7 = ComplexDecoder(4 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
		self.dec8 = ComplexDecoder(3 * B, Mask[0] * Mask[1], kernel_size=(3, 3), stride=(2, 1), padding=(1, 1),
									bias=True)

	def cat(self, x, y, dim):
		real = torch.cat([x.real, y.real], dim)
		imag = torch.cat([x.imag, y.imag], dim)
		return ComplexTensor(real, imag)

	def deepfiltering(self, deepfilter, cplxInput):
		deepfilter = deepfilter.permute(0, 2, 3, 1)
		real_tf_shift = torch.stack(
			[torch.roll(cplxInput.real, (i, j), dims=(1, 2)) for i in self.f_taps for j in self.t_taps], 3).transpose(-1, -2)

		imag_tf_shift = torch.stack(
			[torch.roll(cplxInput.imag, (i, j), dims=(1, 2)) for i in self.f_taps for j in self.t_taps], 3).transpose(-1, -2)

		imag_tf_shift += 1e-10
		cplxInput_shift = ComplexTensor(real_tf_shift, imag_tf_shift)
		est_complex = einsum('bftd,bfdt->bft', [deepfilter.conj(), cplxInput_shift])
		return est_complex

	def forward(self, audio, verbose=False):
		"""
		batch: tensor of shape (batch_size x channels x num_samples)
		"""
		if verbose: print('*' * 60)
		if verbose: print('Input Audio Shape         : ', audio.shape)
		if verbose: print('*' * 60)

		_, _, real, imag = self.stft(audio)
		cplxIn = ComplexTensor(real, imag)
		if verbose: print('STFT Complex Spec         : ', cplxIn.shape)

		if verbose: print('\n' + '-' * 20)
		if verbose: print('Encoder Network')
		if verbose: print('-' * 20)

		enc1 = self.enc1(cplxIn.unsqueeze(1))
		if verbose: print('Encoder-1                 : ', enc1.shape)
		FTB1 = self.FTB1(enc1)
		if verbose: print('FTB-1               : ', FTB1.shape)
		enc2 = self.enc2(FTB1)
		if verbose: print('Encoder-2                 : ', enc2.shape)
		enc3 = self.enc3(enc2)
		if verbose: print('Encoder-3                 : ', enc3.shape)
		enc4 = self.enc4(enc3)
		if verbose: print('Encoder-4                 : ', enc4.shape)
		enc5 = self.enc5(enc4)
		if verbose: print('Encoder-5                 : ', enc5.shape)
		enc6 = self.enc6(enc5)
		if verbose: print('Encoder-6                 : ', enc6.shape)
		enc7 = self.enc7(enc6)
		if verbose: print('Encoder-7                 : ', enc7.shape)
		FTB7 = self.FTB7(enc7)
		if verbose: print('FTB-7               : ', FTB7.shape)
		enc8 = self.enc8(FTB7)
		if verbose: print('Encoder-8                 : ', enc8.shape)

		# +++++++++++++++++++ Expanding Path  +++++++++++++++++++++ #

		MLTB = self.TB(enc8)
		if verbose: print('Transformer-1               : ', MLTB.shape)
		if verbose: print('\n' + '-' * 20)
		if verbose: print('Decoder Network')
		if verbose: print('-' * 20)
		dec = self.dec1(self.cat(MLTB, self.skip1(enc8), 1))
		# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if verbose: print('Decoder-1                 : ', dec.shape)
		dec = self.dec2(self.cat(dec, self.skip2(enc7), 1))
		if verbose: print('Decoder-2                 : ', dec.shape)
		dec = self.dec3(self.cat(dec, self.skip3(enc6), 1))
		if verbose: print('Decoder-3                 : ', dec.shape)
		dec = self.dec4(self.cat(dec, self.skip4(enc5), 1))
		if verbose: print('Decoder-4                 : ', dec.shape)
		dec = self.dec5(self.cat(dec, self.skip5(enc4), 1))
		if verbose: print('Decoder-5                 : ', dec.shape)
		dec = self.dec6(self.cat(dec, self.skip6(enc3), 1))
		if verbose: print('Decoder-6                 : ', dec.shape)
		dec = self.dec7(self.cat(dec, self.skip7(enc2), 1))
		if verbose: print('Decoder-7                 : ', dec.shape)
		dec = self.dec8(self.cat(dec, self.skip8(enc1), 1))
		if verbose: print('Decoder-8                 : ', dec.shape)

		deepfilter = ComplexTensor(dec.real, dec.imag)
		enhanced = self.deepfiltering(deepfilter, cplxIn)
		enh_mag, enh_phase = enhanced.abs(), enhanced.angle()
		audio_enh = self.istft(enh_mag, enh_phase, squeeze=True)
		if verbose: print('*' * 60)
		if verbose: print('Output Audio Shape        : ', audio_enh.shape)
		if verbose: print('*' * 60)

		return audio_enh