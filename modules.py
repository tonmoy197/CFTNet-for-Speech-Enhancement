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

# -------------------------------- Depth wise Seperable Convolution --------------------------------
class depthwise_separable_convx(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_convx, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

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

                #--------------- STEP - 2 : Pointwise Multiplication with input and FTM-----------------
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


# Step : 2 >>>>>>>>>>>>>>>>>>>>>>>>>> Skip Connection >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 3, padding = 1, DSC = False):
        super(SkipBlock, self).__init__()

        if DSC:
            self.conv = DSC_Encoder(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        else:
            self.conv = CplxConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

        self.norm = CplxBatchNorm2d(in_channels)

        def forword(self, x):
            # Q : Why x is added 
            return complex_relu(self.norm(self.conv(x)) + x)

# Final encoder layer has 1 skipblock but first encoder layer has 8 skipblock, 
# quantity of skipblock is inversly proportional to the corresponding encoder 
class SkipConnection(nn.Module):
    def __init__(self, in_channels, num_convblocks, DSC = False):
        self.skip_blocks = [SkipBlock(in_channels, out_channels, kernel_size=3, stride=3, padding=1, DSC=DSC ) for k in range(num_convblocks)]
        self.skip_path = nn.Sequential(*self.skip_blocks)

    def forward(self, x):
        return self.skip_path(x)


# Step : 3 >>>>>>>>>>>>>>>>>>>>>>>> Activation Function >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def complex_mul(x, y ):
    real = x.real * y.real - x.imag * y.imag
    imag = x.real * y.imag + x.imag * y.real
    out = ComplexTensor(real, imag)

def complex_sigmoid(input):
    return F.sigmoid(input.real).type(torch.complex64) + 1j * F.sigmoid(input.imag).type(torch.complex64)

class complex_softplus(nn.Module):
    def __init__(self):
        super(complex_softplus, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, input):
        return self.softplus(input.real).type(complex64) + 1j * self.softplus(input.imag).type(complex64)

class complex_elu(nn.Module):
    def __init__(self):
        super(complex_elu, self).__init__()
        self.elu = nn.ELU(inplace=False)

    def forward(self, input):
        return self.elu(input.real).type(torch.complex64) + 1j * self.elu(input.imag).type(complex64)


# Step : 4 >>>>>>>>>>>>>>>>>>>>>> Bottleneck Layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step : 4.1 --------------------- GRU Layers -----------------------
class ComplexGRU(nn.Module):
    def __init__(self, input_size, num_layers):
        super(ComplexGRU, self).__init__()
        self.rGRU = nn.Sequential(
            nn.GRU(input_size=input_size, hidden_size= input_size//2, num_layers = num_layers, batch_first=True, bidirectional=True),
            SelectItem(0)
        )

        self.iGRU = nn.Sequential(
            nn.GRU(input_size = input_size, hidden_size=input_size // 2, num_layers=num_layers, batch_first=True, bidirectional=True),
            SelectItem(0)
        )
        self.linear = CplxLinear(input_size, output_size)

    def forward(self, x):
        x = x.transpose(-1, -2).contigous()
        real = self.rGRU(x.real) - self.iGRU(x.imag)
        imag = self.rGRU(x.imag) + self.iGRU(x.real)
        out = self.linear(ComplexTensor(real, imag)).transpose(-1, -2) 
        return out 

# Step : 4.2 --------------------- Complex Transformer Layers -------------------
class Transformer_single(nn.Module):
    def __init__(self, nhead = 8):
        super(Transformer_single, self).__init__()
        self.nhead = nhead

    def forward(self, x):
        b, c, F, T = x.shape
        STB = TransformerEncoderLayer(d_model = F, nhead = self.nhead) # Expected Feature
        STB.to("cuda")
        x = x.permute(1, 0, 3, 2).contigous().view(-1, b * T, F) # [c, b * T, F]
        x = x.to("cuda")
        s = STB(x)
        x = x.view(b, c, F, T)
        return x 
    
class Transformer_multi(nn.Module):
    # d_model = x.shape[3]
    def __init__(self, nhead, layer_num = 2):
        super(Transformer_multi, self).__init__()
        self.layer_num = layer_num
        self.MTB = Transformer_single(nhead=nhead)

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.MTB(x)
        return x

class ComplexTransformer(nn.Module):
    def __init__(self, nhead, num_layer):
        super(ComplexTransformer, self).__init__()
        self.rTrans = Transformer_multi(nhead=nhead, layer_num=num_layer)
        self.iTrans = Transformer_multi(nhead=nhead, layer_num=num_layer)
    
    def forward(self, x):
        real = self.rTrans(x.real) - self.iTrans(x.imag)
        imag = self.rTrans(x.imag) + self.iTrans(x.real)
        out = ComplexTensor(real, imag)
        return out 

class TransformerEncoderLayer(Module):
    '''
    Args : 
        d_model : the number of expected features in the input (required )
        nhead : the number of heads in the multiheadattention model (Required)
        dim_feedforward : the dimension of the feedforward network model (default = 2048)
        dropout : the dropout value (default = 0.1)
        activation : the activation function of intermediate layer, relu or elu (default = relu)

    Example : 
        encoder_layer = nn.TransformerEncoderLayer(d_model = 512, nhead = 8)
        src = torch.rand(10, 32, 512)
        out = encoder_layer(src)
    '''
    def __init__(self, d_model, nhead, bidirectional = True, dropout = 0, activation = "relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout).to("cuda")

        #----------------Feedforward Model ------------------
        self.gru = GRU(d_model, dim_feedforward * 2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)

        if bidirectional:
            self.linear2 = Linear(d_model * 2 * 2, d_model)
        
        else : 
            self.linear2 = Linear(d_model * 2 , d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_mask = None, src_key_padding_mask = None):
        """ 
        Args : 
            src : the sequence to the encoder layer (required)
            src_mask : the mask for the src sequence (optional )
            src_key_padding_mask : the mask for the src key per batch (optional)
        """

        src2 = self.self_attn(src, src, src, attn_mask = src_mask, src_key_padding_mask = src_key_padding_mask)
        src = src + self.dropout(src2)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(module, N):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


