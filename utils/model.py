import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import scipy.io as scio
import os
import time
import torch.nn.functional as F
import sys

# sys.path.append('/home/yangbo_zhou/HRRP/transformer_2024/decompose_loss_gate')

from utils.attn import CrossAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)   
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  
        return x
    


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V,d_k): 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, enc_inputs):
        residual, batch_size = enc_inputs, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        context, attn = ScaledDotProductAttention()(Q, K, V,self.d_k)
        context = context.transpose(1, 2).reshape( batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context) 
        return output + residual

class Token_module(nn.Module):  
    def __init__(self,seq_len):
        super(Token_module, self).__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=seq_len, eps=1e-5, affine=True)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=seq_len, eps=1e-5, affine=True)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool1d(kernel_size=2)

    def forward(self, enc_inputs):
        output = self.conv1(enc_inputs)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.pooling1(output)

        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = self.relu2(output)
        output = self.pooling2(output)
        
        
        return output
    


class Local_module(nn.Module):
    def __init__(self):
        super(Local_module, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=1, eps=1e-5, affine=True)
        self.relu2 = nn.ReLU()


        self.conv_res = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.batchnorm_res = nn.BatchNorm2d(num_features=1, eps=1e-5, affine=True)

        self.rule = nn.ReLU()
    def forward(self, enc_inputs): 
        enc_inputs = enc_inputs.unsqueeze(1)
        output = self.conv1(enc_inputs)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = self.relu2(output)

        output_res = self.conv_res(enc_inputs)
        output_res = self.batchnorm_res(output_res)

        output = self.rule(output+output_res)

        return output.squeeze(1)
    

class Global_module(nn.Module):
    def __init__(self,d_model,d_ff,d_k,d_v,n_heads):
        super(Global_module, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layernorm1 = nn.LayerNorm(normalized_shape=[d_model], eps=1e-05, elementwise_affine=True)
        self.encoder_layers = MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[d_model], eps=1e-05, elementwise_affine=True)
        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.activate = nn.GELU()
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
    def forward(self, enc_inputs):
        X_gin = self.layernorm1(enc_inputs)
        X_gmout = self.encoder_layers(X_gin, X_gin, X_gin, enc_inputs)
        X_out = self.layernorm2(X_gmout)
        X_out = self.ffn1(X_out)
        X_out = self.activate(X_out)
        X_out = self.ffn2(X_out) + X_gmout
        return X_out
    
class FB_Unit(nn.Module):
    def __init__(self,seq_len):
        super(FB_Unit, self).__init__()
        self.seq_len = seq_len
        self.conv = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=1)
    def forward(self, enc_inputs):
        output = self.conv(enc_inputs)
        return output
    

class GatedFusion(nn.Module):
    def __init__(self, d_model,size_out=16):
        super(GatedFusion, self).__init__()
        self.d_model = d_model
        self.size_out = size_out
        self.hidden1 = nn.Linear(d_model, size_out, bias=False)
        self.hidden2 = nn.Linear(d_model, size_out, bias=False)     
        self.hidden_sigmoid  = nn.Linear(size_out * 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, feature1, feature2):
        hide1 = self.tanh(self.hidden1(feature1))
        hide2 = self.tanh(self.hidden2(feature2))
        combined = torch.cat((hide1,hide2),dim=2)
        gate = self.sigmoid(self.hidden_sigmoid(combined))
        fused = torch.mean(gate.view(-1))*hide1 + torch.mean((1-gate).view(-1))*hide2
        return fused

class MyLayer_single(nn.Module):
    def __init__(self, batch_size, d_model,n_layers_L,n_layers_G,tgt_len,device,d_ff,seq_len,d_k,d_v,n_heads):
        super(MyLayer_single, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_layers_L = n_layers_L
        self.n_layers_G = n_layers_G
        self.tgt_len = tgt_len
        self.device = device
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.S2T = Token_module(seq_len=seq_len)
        self.posi_enc = PositionalEncoding(d_model=d_model)
        self.layers_Local = nn.ModuleList([Local_module() for _ in range(n_layers_L)])
        self.layers_Global = nn.ModuleList([Global_module(d_model=d_model,d_ff=d_ff,d_k=d_k,d_v=d_v,n_heads=n_heads) for _ in range(n_layers_G)])
        self.layers_FB_unit = nn.ModuleList([FB_Unit(seq_len=seq_len) for _ in range(10)])

    def forward(self, enc_inputs):
        enc_inputs = self.S2T(enc_inputs)
        bs, _, model_dim = enc_inputs.shape
        x1 = nn.Parameter(torch.randn(bs, 1, model_dim).to(self.device)).to(enc_inputs.dtype)
        enc_inputs_cls = torch.cat([x1, enc_inputs], dim=1)
        enc_inputs_posi = self.posi_enc(enc_inputs_cls)
        for layer_G_TSFE, layer_L_SFE, layer_FB_unit in zip(self.layers_Global, self.layers_Local, self.layers_FB_unit):
            enc_inputs_global = layer_G_TSFE(enc_inputs_posi)
            enc_inputs = layer_L_SFE(enc_inputs)
            enc_inputs_fb = layer_FB_unit(enc_inputs)
            enc_inputs_posi = torch.cat([enc_inputs_global[:,0,:].unsqueeze(1), enc_inputs_global[:,1:,:]+enc_inputs_fb], dim=1)
        global_futrue = enc_inputs_global[:,1:,:]
        detail_futrue = enc_inputs_fb
        return global_futrue,detail_futrue

class MyLayer(nn.Module):
    def __init__(self, batch_size, d_model, n_layers_L,n_layers_G, tgt_len, device, d_ff, seq_len, d_k, d_v, n_heads,size_out):
        super(MyLayer, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_layers_L = n_layers_L
        self.n_layers_G = n_layers_G
        self.tgt_len = tgt_len
        self.device = device
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.size_out= size_out

        self.layer_eh = MyLayer_single(batch_size, d_model, n_layers_L,n_layers_G, tgt_len, device, d_ff, seq_len, d_k, d_v, n_heads)
        self.layer_ev = MyLayer_single(batch_size, d_model, n_layers_L,n_layers_G, tgt_len, device, d_ff, seq_len, d_k, d_v, n_heads)
        self.gated_fusion_global = GatedFusion(d_model,size_out)
        self.gated_fusion_local = GatedFusion(d_model,size_out)
        self.cross_attn = CrossAttention(size_out, d_k, d_v, n_heads)
        self.fc = nn.Linear(size_out, tgt_len, bias=True)

    def forward(self, inputs_eh, inputs_ev):
        global_futrue_eh, detail_futrue_eh = self.layer_eh(inputs_eh)
        global_futrue_ev, detail_futrue_ev = self.layer_ev(inputs_ev)

        global_fused = self.gated_fusion_global(global_futrue_eh, global_futrue_ev)
        detail_fused = self.gated_fusion_local(detail_futrue_eh, detail_futrue_ev)

        final_fused,_ = self.cross_attn(global_fused, detail_fused)
        output = self.fc(final_fused[:, 0, :].squeeze(-1))

        return output, global_futrue_eh, detail_futrue_eh, global_futrue_ev, detail_futrue_ev