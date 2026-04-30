# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:57:57 2022

@author: LSH
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
import utils
from loss import ZINBLoss, contrastive_loss
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
import time
import random
import scipy
import scipy.sparse

random.seed(1)

  # 它的作用是训练 AttentionAE 模型，同时优化输入数据的重构（通过 ZINB 损失） 和图结构的重构（通过 MSE 损失）
def train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args):
    # 定义训练函数，参数包括：初始模型、标准化数据、原始数据、邻接矩阵、重构目标邻接矩阵、size factor
    # 记录训练开始时间
    start_time = time.time()
    # 记录训练开始时的GPU最大内存占用（用于计算训练过程中的内存消耗）
    start_mem = torch.cuda.max_memory_allocated(device=device)
    # 将模型移动到指定计算设备（CPU或GPU）
    init_model.to(device)
    # 将标准化数据转换为PyTorch张量并移动到指定设备
    data = torch.Tensor(Zscore_data).to(device)
    # 将size factor转换为张量，包装为可自动求导的变量，并移动到指定设备（size factor用于ZINB模型的缩放）单细胞分析常用的规模因子，用于对均值 mean 做缩放，防止测序深度的差异影响
    sf = torch.autograd.Variable((torch.from_numpy(size_factor[:,None]).type(torch.FloatTensor)).to(device),
                           requires_grad=True)
    optimizer = torch.optim.Adam(init_model.parameters(), lr=args.lr)  # 初始化Adam优化器，学习率从超参数获取
    # 处理邻接矩阵：如果是scipy的稀疏矩阵格式，转换为PyTorch稀疏张量；否则直接转换为张量并移动到设备
    if type(adj) ==scipy.sparse._csr.csr_matrix:
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(device)  # 调用工具函数转换稀疏矩阵
        r_adj = torch.Tensor(r_adj.toarray()).to(device)     # 将重构目标邻接矩阵转换为稠密张量
    else:
        adj = torch.Tensor(adj).to(device)    # 直接转换为张量
        r_adj = torch.Tensor(r_adj).to(device)   # 直接转换为张量
        # 初始化学习率调度器：每40个epoch将学习率乘以0.5（gamma），last_epoch=-1表示从头开始
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, last_epoch=-1)
    best_model = init_model    # 初始化最佳模型为初始模型
    loss_update = 100000       # 初始化最佳损失为一个较大值，用于记录最小损失
    for epoch in range(args.training_epoch):
        # 1. 获取模型输出 (注意接收新增的两个返回值)
        z, A_pred, pi, mean, disp, z_dae_proj, z_gnn_proj = init_model(data, adj)
        
        l = ZINBLoss(theta_shape=(args.n_input,))
        zinb_loss = l(mean * sf, pi, target=torch.tensor(rawData).to(device), theta=disp)
        
        re_graphloss = torch.nn.functional.mse_loss(A_pred.view(-1), r_adj.view(-1))

        # 温度参数 temperature 一般设为 0.5 或 0.1
        con_loss = contrastive_loss(z_dae_proj, z_gnn_proj, temperature=0.5)
        
        # 2. 计算总损失 (给对比损失一个权重，例如 0.1)
        # Loss = ZINB + 0.1 * Graph_Recon + 0.1 * Contrastive
        loss = zinb_loss + 0.1 * re_graphloss + 0.1 * con_loss

        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.4f, zinb_loss %.4f, re_graphloss %.4f, con_loss %.4f" % (
            epoch + 1, loss, zinb_loss, re_graphloss, con_loss))

        if loss_update > loss:
            loss_update = loss    # 如果当前损失小于历史最佳损失，更新最佳损失和最佳模型
            best_model = init_model
            epoch_update = epoch  # 记录最佳模型对应的epoch
            # 早停机制：如果连续50个epoch损失未更新，停止训练
        if ((epoch - epoch_update) > 50):
            print("Early stopping at epoch {}".format(epoch_update))
            elapsed_time = time.time() - start_time   # 计算训练耗时
            # 计算最大内存消耗（当前最大占用 - 初始占用），转换为MB
            max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
            print("Finish Training! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
            return best_model 
        optimizer.zero_grad()  # 清空梯度，避免累积
        loss.backward()      # 反向传播计算梯度
        # 梯度裁剪：限制梯度的L2范数最大为3，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(init_model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()     # 优化器更新参数
        scheduler.step()    # 学习率调度器更新学习率
        # 训练结束（未触发早停），计算耗时和内存消耗
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
    print("Finish Training! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
    return best_model, elapsed_time   # 返回最佳模型和训练耗时
    
alpha = 1    # 定义聚类损失中的温度参数alpha


# 定义聚类损失函数（基于KL散度），输入：潜在变量z、聚类中心cluster_layer
def loss_func(z, cluster_layer):
    # 计算q分布：衡量z与聚类中心的相似度（基于Student's t分布）
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - cluster_layer) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0    # 对相似度进行幂运算
    q = (q.t() / torch.sum(q, dim=1)).t()       # 行归一化，得到概率分布
    # 计算目标分布p：基于q的平方归一化，使高置信度的样本权重更大
    weight = q ** 2 / torch.sum(q, dim=0)     # 计算权重
    p = (weight.t() / torch.sum(weight, dim=1)).t()     # 行归一化，得到目标分布p

    log_q = torch.log(q)     # 计算q的对数（用于KL散度）
    # 计算q和p的KL散度作为损失（批量平均）
    loss = torch.nn.functional.kl_div(log_q, p, reduction='batchmean')
    return loss, p   # 返回KL损失和目标分布p


# 定义聚类函数，参数包括：预训练模型、标准化数据、原始数据、细胞类型标签、邻接矩阵、重构目标邻接矩阵、size factor、设备、超参数
def clustering(pretrain_model, Zscore_data, rawData, celltype, adj, r_adj, size_factor, device, args):
    start_time = time.time()    # 记录聚类开始时间
    
    start_mem = torch.cuda.max_memory_allocated(device=device)    # 记录聚类开始时的GPU内存占用
    
    data = torch.Tensor(Zscore_data).to(device)   # 转换标准化数据为张量并移动到设备
    # 处理邻接矩阵（同train函数）
    if type(adj) ==scipy.sparse._csr.csr_matrix:
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(device)
        r_adj = torch.Tensor(r_adj.toarray()).to(device)
    else:
        adj = torch.Tensor(adj).to(device)
        r_adj = torch.Tensor(r_adj).to(device)
        
    model = pretrain_model.to(device)     # 将预训练模型移动到指定设备
    # 处理size factor（同train函数）
    sf = torch.autograd.Variable((torch.from_numpy(size_factor[:,None]).type(torch.FloatTensor)).to(device),
                          requires_grad=True)
    #cluster center
    # 聚类中心初始化：用预训练模型得到潜在变量z（不计算梯度，节省资源）
    with torch.no_grad():
        z, _, _, _, _, _, _ = model(data, adj)
        # 调用Leiden算法对z进行聚类，得到初始聚类中心和标签（Leiden是一种高效的社区检测算法）
    cluster_centers, init_label = utils.use_Leiden(z.detach().cpu().numpy(), resolution=args.resolution)
    # 将聚类中心转换为可学习的参数（允许梯度更新）并移动到设备
    cluster_layer = torch.autograd.Variable((torch.from_numpy(cluster_centers).type(torch.FloatTensor)).to(device),
                           requires_grad=True)
    # 计算初始轮廓系数（ASW，衡量聚类紧凑性和分离度）
    asw = np.round(silhouette_score(z.detach().cpu().numpy(), init_label), 3)
    if celltype is not None:  # 如果提供了真实细胞类型标签，计算NMI和ARI
        nmi = np.round(normalized_mutual_info_score(celltype, init_label), 3)   # 标准化互信息
        ari = np.round(adjusted_rand_score(celltype, init_label), 3)            # 调整兰德指数
        print('init: ASW= %.3f, ARI= %.3f, NMI= %.3f' % (asw, ari, nmi)) 
    else:    # 无真实标签时只打印ASW
        print('init: ASW= %.3f' % (asw))
        # 初始化优化器：优化模型的编码器、注意力机制、GNN层、潜在变量层参数，以及聚类中心
    optimizer = torch.optim.Adam(list(model.enc_1.parameters()) + list(model.enc_2.parameters()) + 
                                  list(model.attn1.parameters()) + list(model.attn2.parameters()) + 
                                  list(model.gnn_1.parameters()) + list(model.gnn_2.parameters()) +
                                 list(model.z_layer.parameters()) + [cluster_layer], lr=0.001)   
    
    for epoch in range(args.clustering_epoch):    # 循环聚类训练指定的epoch次数
        # 前向传播得到模型输出
        z, A_pred, pi, mean, disp, _, _ = model(data, adj)
        kl_loss, ae_p = loss_func(z, cluster_layer)   # 计算KL损失和目标分布p
        l = ZINBLoss(theta_shape=(args.n_input,))     # 初始化ZINB损失
        zinb_loss = l(mean * sf, pi, target=torch.tensor(rawData).to(device), theta=disp)    # 计算ZINB损失
        re_graphloss = torch.nn.functional.mse_loss(A_pred.view(-1), r_adj.to(device).view(-1))   # 计算图重构损失
        # 总损失 = KL损失 + 0.1*ZINB损失 + 0.01*图重构损失（不同权重平衡）
        loss = kl_loss + 0.1 * zinb_loss + 0.01*re_graphloss
        loss.requires_grad_(True)   # 确保损失可求导
        label = utils.dist_2_label(ae_p)   # 从分布p中获取预测标签（如取最大概率对应的类别）
        # 计算当前的轮廓系数和戴维斯-布尔丁指数（DB，值越小聚类越好）
        asw = silhouette_score(z.detach().cpu().numpy(), label)
        db = davies_bouldin_score(z.detach().cpu().numpy(), label)
        # ari = adjusted_rand_score(celltype, label)
        # nmi = normalized_mutual_info_score(celltype, label)
       

        if (epoch+1) % 10 == 0:    # 每10个epoch打印损失和ASW
            print("epoch %d, loss %.4f, kl_loss %.4f, ASW %.3f"% (epoch+1, loss, kl_loss, asw))
        num = data.shape[0]   # 样本数量（细胞数量）
        tol=1e-3    # 标签变化容忍阈值（早停条件）
        if epoch == 0:   # 记录第一个epoch的标签
            last_label = label
        else:       # 从第二个epoch开始计算标签变化率
            delta_label = np.sum(label != last_label).astype(np.float32) / num   # 标签变化比例
            last_label = label    # 更新上一轮标签
            # 早停条件：epoch>20且标签变化比例小于阈值
            if epoch>20 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                elapsed_time = time.time() - start_time     # 计算聚类耗时
                max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem  # 计算内存消耗
                print("Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
                break  
        optimizer.zero_grad()   # 清空梯度
        loss.backward()     # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)  # 梯度裁剪
        optimizer.step()   # 更新参数
        # 聚类结束（未触发早停），计算耗时和内存消耗
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
    print("Finish Clustering! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time, max_mem / 1024 / 1024))
    # 返回聚类评估指标（ASW、DB）、最终标签、聚类中心、训练好的模型、
    return [asw,db], last_label, cluster_layer, model, elapsed_time

