# -*- coding: utf-8 -*-
# DGC-MSA: Dual-View Graph Contrastive Clustering via Multi-Scale Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
# Three different activation function uses in the ZINB-based denoising autoencoder.
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e4)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e3)
PiAct = lambda x: 1/(1+torch.exp(-1 * x))

# A general GCN layer.
 # 图神经网络层
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = torch.tanh(output)
        return output
class MultiScaleGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, scales=[1, 2, 3]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.out_features = out_features

        self.gnn_layers = nn.ModuleList([
            GNNLayer(in_features, out_features) for _ in scales
        ])

        self.scale_attention = nn.Sequential(
            nn.Linear(out_features * self.num_scales, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_scales),
            nn.Softmax(dim=-1)
        )

        self.output_proj = nn.Linear(out_features, out_features)

        self.norm = nn.LayerNorm(out_features) 
        
    def propagate_multiscale(self, features, adj, scale):
        if scale == 1:
            adj_scale = adj
        else:
            adj_scale = torch.matrix_power(adj.float(), scale)
            adj_scale = (adj_scale > 0).float()
        return adj_scale
    
    def forward(self, features, adj):
        scale_outputs = []
        for i, scale in enumerate(self.scales):
            adj_scale = self.propagate_multiscale(features, adj, scale)
            scale_out = self.gnn_layers[i](features, adj_scale, active=True)
            scale_outputs.append(scale_out)
        concat_features = torch.cat(scale_outputs, dim=-1)
        global_features = concat_features.mean(dim=0, keepdim=True) 
        attention_weights = self.scale_attention(global_features)
        weighted_output = torch.zeros_like(scale_outputs[0])
        
        for i in range(self.num_scales):
            weight = attention_weights[:, i].view(1, 1) 
            weighted_output += weight * scale_outputs[i]
        output = self.output_proj(weighted_output)
        output = self.norm(output)
        
        return output

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

class GaussianNoise(nn.Module):
    def __init__(self, device, sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device = device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
        return x

# A multi-head attention layer has two different input (query and key/value).
class AttentionWide(nn.Module):
    def __init__(self, emb, p=0.2, heads=8, use_gate=True):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.use_gate = use_gate
        self.dropout = nn.Dropout(p)

        # 定义 Q, K, V 映射
        # Query 来自 GNN，Key/Value 来自 DAE
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        # 输出映射
        self.unifyheads = nn.Linear(heads * emb, emb)

        # LayerNorm
        self.norm = nn.LayerNorm(emb)

        # 门控机制
        if self.use_gate:
            self.gate_layer = nn.Sequential(
                nn.Linear(2 * emb, emb),
                nn.Sigmoid()
            )

    def forward(self, x, y):
        # x: DAE 特征 (Key/Value) [N, emb]
        # y: GNN 特征 (Query)     [N, emb]
        
        residual = x 

        b = 1 
        t, e = x.size() # x 是 2维 [N, emb]
        h = self.heads

        # 生成 Q, K, V
        queries = self.dropout(self.toqueries(y)).view(b, t, h, e)
        keys = self.tokeys(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # 维度变换
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # Attention 计算
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        # 加权求和
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        
        # 映射回原始维度 -> 得到 3维张量 [1, N, emb]
        attn_out = self.unifyheads(out)
        # 将 3维 [1, N, emb] 压缩为 2维 [N, emb]，以便与 residual 拼接
        attn_out = attn_out.squeeze(0)

        # 门控融合
        if self.use_gate:
            # 现在 residual 和 attn_out 都是 2维 [N, emb]，可以拼接了
            combined = torch.cat([residual, attn_out], dim=-1)
            gate = self.gate_layer(combined)
            output = residual + gate * attn_out
        else:
            output = residual + attn_out
        
        # LayerNorm
        output = self.norm(output)

        return output
# Final model
class DGC_MSA(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, heads, device, use_multiscale_gnn=True,
                 gnn_scales=[1, 2, 3]):
        super(DGC_MSA, self).__init__()
        self.use_multiscale_gnn = use_multiscale_gnn
        self.gnn_scales = gnn_scales
        self.Gnoise = GaussianNoise(sigma=0.1, device=device)

        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN_1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN_2 = nn.BatchNorm1d(n_enc_2)
        self.z_layer = nn.Linear(n_enc_2, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.calcu_pi = nn.Linear(n_dec_2, n_input)
        self.calcu_disp = nn.Linear(n_dec_2, n_input)
        self.calcu_mean = nn.Linear(n_dec_2, n_input)
        
        # --- GNN 层 ---
        if use_multiscale_gnn:
            self.gnn_1 = MultiScaleGNNLayer(n_enc_1, n_enc_2, scales=gnn_scales)
            self.gnn_2 = MultiScaleGNNLayer(n_enc_2, n_z, scales=gnn_scales)
        else:
            self.gnn_1 = GNNLayer(n_enc_1, n_enc_2)
            self.gnn_2 = GNNLayer(n_enc_2, n_z)
            
        # --- 注意力层 ---
        self.attn1 = AttentionWide(n_enc_2, heads=heads, use_gate=True)
        self.attn2 = AttentionWide(n_z, heads=heads, use_gate=True)

        # 将潜在特征映射到对比空间，通常使用 MLP (Linear->ReLU->Linear)
        self.proj_head = nn.Sequential(
            nn.Linear(n_z, n_z),
            nn.ReLU(),
            nn.Linear(n_z, n_z)
        )

    def forward(self, x, adj):
        # 1. 编码器路径第一层
        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        
        # 2. GNN路径第一层
        h1 = self.gnn_1(enc_h1, adj)
        
        # 3. GNN路径第二层 (GNN View Feature)
        h2 = self.gnn_2(h1, adj) 
        
        # 4. 编码器路径第二层 & 第一次融合
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        enc_h2 = self.attn1(enc_h2, h1)
        
        # 5. 生成潜在变量 (DAE View Feature - 融合前)
        # 注意：这里我们提取 z_pre 作为 DAE 视图的特征，因为它还没融合 h2
        z_pre = self.z_layer(self.Gnoise(enc_h2))

        # 将 DAE 视图 (z_pre) 和 GNN 视图 (h2) 映射到对比空间
        z_dae_proj = self.proj_head(z_pre)
        z_gnn_proj = self.proj_head(h2)
        
        # 6. 第二次融合 (得到最终用于聚类的 z)
        z = self.attn2(z_pre, h2)
        
        # 7. 解码器路径
        A_pred = dot_product_decode(h2) # 预测图结构
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        
        pi = PiAct(self.calcu_pi(dec_h2))
        mean = MeanAct(self.calcu_mean(dec_h2))
        disp = DispAct(self.calcu_disp(dec_h2))
        
        # 返回值增加了两个投影特征
        return z, A_pred, pi, mean, disp, z_dae_proj, z_gnn_proj
