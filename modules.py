import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from cplxmodule.nn import CplxConv2d, CplxConvTranspose2d, CplxBatchNorm2d, CplxLinear, CplxConv1d, CplxBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(9999)
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parameters()])
    return np.round(neles / 10 ** 6 if Mb else neles, 2)


# -----------------------   Architecture Parameters --------------------------------

# Step: 1.1 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Encoder/Decoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class ComplexEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False,
                 DSC=False):
        super(ComplexEncoder, self).__init__()
        # DSC: depthwise_separable_conv
        if DSC:
            self.conv = DSC_Encoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
        else:
            self.conv = CplxConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=bias)
        self.norm = CplxBatchNorm2d(out_channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x)))


class ComplexDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 output_padding=(0, 0), bias=False, DSC=False):
        super(ComplexDecoder, self).__init__()
        # DSC: depthwise_separable_conv
        if DSC:
            self.conv = DSC_Decoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    output_padding=output_padding, bias=bias)
        else:
            self.conv = CplxConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, bias=bias)

        self.norm = CplxBatchNorm2d(out_channels)

    def forward(self, x):
        return complex_relu(self.norm(self.conv(x)))


class DSC_Encoder(nn.Module):
    # depthwise_separable_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False):
        super(DSC_Encoder, self).__init__()
        self.depthwise = CplxConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    groups=in_channels, bias=bias)  # group = in_ch; and in_ch=out_ch
        self.pointwise = CplxConv2d(in_channels, out_channels, kernel_size=1, bias=bias)  # Kernel_size = 1 always

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DSC_Decoder(nn.Module):
    # depthwise_separable_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=(1, 1), output_padding=(0, 0),
                 bias=False):
        super(DSC_Decoder, self).__init__()
        self.depthwise = CplxConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                             padding=padding, groups=in_channels, output_padding=output_padding,
                                             bias=bias)  # group = in_ch; and in_ch=out_ch
        self.pointwise = CplxConvTranspose2d(in_channels, out_channels, kernel_size=1,
                                             bias=bias)  # Kernel_size = 1 always

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# Step: 1.2 >>>>>>>>>>>>>>>>>>>>>>>>>>>  Frequency Tranformation Block >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Change the shape of the feature
class NodeReshape(nn.Module):
    def __init__(self, shape):
        super(NodeReshape, self).__init__()
        self.shape = shape

    def forward(self, feature_in : torch.Tensor):
        # Finding the desired shape
        shape = feature_in.size()
        batch = shape[0]
        new_shape = [batch]
        new_shape.extend(list(self.shape))
        # Reshaping 
        return feature_in.reshape(new_shape)
    
# performs a complex linear transformation on the input tensor
class Freq_FC(nn.Module):
    def __init__(self, F_dim, bias = False):
        super(Freq_FC, self).__init__()
        self.linear = CplxLinear(F_dim, F_dim, bias = bias)

    def forward(self, x):
        out = x.transpose(-1, -2).contiguous() # [Batch , Channel_in_out , T, F]
        # Apply the complex linear layer
        out = self.linear(out) # .contigous()
        # Convert the real and imaginary parts to a complex tensor
        out = torch.complex(out.real, out.img)
        # Transpose the last two dimensions back to the original order and ensure contiguous memory layout;
        out = out.transpose(-1, -2).contiguous() # [Batch, channel_in_out , F, T]
        return out 


class ComplexFTB(torch.nn.Module):

    def __init__(self, F_dim, channels):
        super(ComplexFTB, self).__init__()
        self.channels = channels
        self.C_r = 5
        self.F_dim = F_dim

        # Presumably a complex-valued 2D convolutional layer and batch normalization
        self.Conv1D_1 = nn.Sequential(
            CplxConv1d(self.F_dim * self.C_r, self.F_dim, kernel_size =  1, stride = 1, padding = 0),
            CplxBatchNorm1d(self.C_r)
        )
        self.Conv2D_1 = nn.Sequential(
            CplxConv2d(in_channels=self.channels, out_channels=self.C_r, kernel_size=1, stride=1, padding=0),
            CplxBatchNorm2d(self.C_r),

        )
        self.FC = Freq_FC(self.F_dim, bias=False)
        self.Conv2D_2 = nn.Sequential(
            CplxConv2d(2*self.channels, self.channels, kernel_size = 1, stride = 1, padding = 0),
            CplxBatchNorm2d(self.channels)
        )
        self.att_inner_reshape = NodeReshape([self.F_dim * self.C_r, -1])
        self.att_out_reshape = NodeReshape([1, F_dim, -1])

        def cat(self, x, y, dim):
            real = torch.cat([x.real, y.real], dim)
            imag = torch.cat([x.imag, y.imag], dim)
            return ComplexTensor(real, imag)
        
        def forward(self, inputs, verbose=False):
                # feature_n: [batch, channel_in_out, T, F]
                _, _, self.F_dim, self.T_dim = inputs.shape

                #-------------- STEP - 1 : T-F Attention Module ------------------------------
                # Conv2D 
                out = complex_relu(self.Conv2D_1(inputs));
                if verbose: print('Layer-1               : ', out.shape)  # [B,Cr,T,F]
                # Reshape: [batch, channel_attention, F, T] -> [batch, channel_attention*F, T]
                out = out.view(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])
                # out = self.att_inner_reshape(out);
                if verbose: print('Layer-2               : ', out.shape)
                # out = out.view(-1, self.T_dim, self.F_dim * self.C_r) ; print(out.shape) # [B,c_ftb_r*f,segment_length]
                # Conv1D
                out = complex_relu(self.Conv1D_1(out));
                if verbose: print('Layer-3               : ', out.shape)  # [B,F, T]
                # temp = self.att_inner_reshape(temp); print(temp.shape)
                out = out.unsqueeze(1)
                # out = out.view(-1, self.channels, self.F_dim, self.T_dim);
                if verbose: print('Layer-4               : ', out.shape)  # [B,c_a,segment_length,1]

                #--------------- STEP -2 : Pointwise Multiplication with input and FTM-----------------
                out = out * inputs;
                if verbose: print('Layer-5               : ', out.shape)  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]
                # Frequency- FC
                # out = torch.transpose(out, 2, 3)  # [batch, channel_in_out, T, F]
                out = self.FC(out);
                # if verbose: print('Layer-6               : ', out.shape)  # [B,c_a,segment_length,f]
                # out = torch.transpose(out, 2, 3)  # [batch, channel_in_out, T, F]
                
                #---------------- STEP -3 : Concatenation with Input and Conv2D ---------------------------
                out = self.cat(out, inputs, 1);
                if verbose: print('Layer-7               : ', out.shape)  # [B,2*c_a,segment_length,f]
                # Conv2D
                outputs = complex_relu(self.Conv2D_2(out));
                if verbose: print('Layer-8               : ', outputs.shape)  # [B,c_a,segment_length,f]

                return outputs

