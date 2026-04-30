# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:34:50 2022

@author: LSH
"""
import torch
from warnings import simplefilter 
import argparse
from sklearn import preprocessing
import random
import numpy as np
import utils
import scipy
import scipy.sparse

from model import AttentionAE
from train import train, clustering, loss_func
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
if __name__ == "__main__":
    # 创建命令行参数解析器，description为脚本描述，formatter_class设置参数帮助信息格式
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 定义学习率参数，类型为float，默认值0.001，帮助信息说明其作用
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate, default:0.001')
    # 定义潜在向量维度参数，类型为int，默认16，描述为每个细胞的潜在向量维度
    parser.add_argument('--n_z', type=int, default=16, 
                        help='the number of dimension of latent vectors for each cell, default:16')
    # 定义注意力头数参数，默认8，用于注意力机制的多头注意力
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='the number of dattention heads, default:8')
    # 定义高变基因数量参数，默认2500，单细胞数据预处理中常用高变基因筛选
    parser.add_argument('--n_hvg', type=int, default=2500, 
                        help='the number of the highly variable genes, default:2500')
    # 定义预训练轮数参数，默认200，即自编码器的训练轮数
    parser.add_argument('--training_epoch', type=int, default=200,
                        help='epoch of train stage, default:200')
    # 定义聚类阶段轮数参数，默认100，即结合聚类损失的训练轮数
    parser.add_argument('--clustering_epoch', type=int, default=100,
                        help='epoch of clustering stage, default:100')
    # 定义Leiden聚类分辨率参数，默认1.0，控制聚类数量（值越小聚类越多）
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='''the resolution of Leiden. The smaller the settings to get the more clusters
                        , advised to 0.1-1.0, default:1.0''')
    # 定义细胞连接性构建方法参数，默认'gauss'（高斯核），可选'umap'
    parser.add_argument('--connectivity_methods', type=str, default='gauss',
                        help='method for constructing the cell connectivity ("gauss" or "umap"), default:gauss')
    # 定义近邻数量参数，默认15，用于构建细胞相似性图时的邻域大小
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='''The size of local neighborhood (in terms of number of neighboring data points) used 
                        for manifold approximation. Larger values result in more global views of the manifold, while 
                        smaller values result in more local data being preserved. In general values should be in the 
                        range 2 to 100. default:15''')
    # 定义是否使用KNN图参数，默认False（用高斯核），True则限制近邻数为n_neighbors
    parser.add_argument('--knn', type=int, default=False,
                        help='''If True, use a hard threshold to restrict the number of neighbors to n_neighbors, 
                        that is, consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to
                        neighbors more distant than the n_neighbors nearest neighbor. default:False''')
    # 定义输入文件名参数，默认'Quake_10x_Spleen'，输入为h5ad格式（含原始计数矩阵X）
    parser.add_argument('--name', type=str, default='Romanov',
                        help='name of input file(a h5ad file: Contains the raw count matrix "X")')
    # 定义细胞类型标签是否已知参数，默认'known'（真实标签在adata.obs["celltype"]中）
    parser.add_argument('--celltype', type=str, default='known',
                        help='the true labels of datasets are placed in adata.obs["celltype"]')
    # 定义是否保存预测标签参数，默认False，保存路径为"./pred_label"
    parser.add_argument('--save_pred_label', type=str, default=False,
                        help='To choose whether saves the pred_label to the dict "./pred_label"')
    # 定义是否保存模型参数参数，默认False，保存路径为"./model_save"
    parser.add_argument('--save_model_para', type=str, default=False,
                        help='To choose whether saves the model parameters to the dict "./model_save"')
    # 定义是否保存细胞嵌入向量参数，默认False，保存路径为"./embedding"
    parser.add_argument('--save_embedding', type=str, default=False,
                        help='To choose whether saves the cell embedding to the dict "./embedding"')
    # 定义是否保存UMAP可视化结果参数，默认False，保存路径为"./umap_figure"
    parser.add_argument('--save_umap', type=str, default=False,
                        help='To choose whether saves the visualization to the dict "./umap_figure"')
    # 定义最大细胞数阈值参数，默认4000，超过则下采样（8GB GPU约可处理4000细胞）
    parser.add_argument('--max_num_cell', type=int, default=4000,
                        help='''a maximum threshold about the number of cells use in the model building, 
                        4,000 is the maximum cells that a GPU owning 8 GB memory can handle. 
                        More cells will bemploy the down-sampling straegy, 
                        which has been shown to be equally effective,
                        but it's recommended to process data with less than 24,000 cells at a time''')
    # 定义是否使用GPU参数，默认False（用CPU）
    parser.add_argument('--cuda', type=bool, default=False,
                        help='use GPU, or else use cpu (setting as "False")')
    args = parser.parse_args()     # 解析命令行参数，得到参数对象args
    device = torch.device("cuda" if args.cuda else "cpu")    # 根据--cuda参数选择计算设备（GPU或CPU）
    simplefilter(action='ignore', category=FutureWarning)
    
    random.seed(1)
    # 如果需要保存UMAP图像，设置保存路径（预测标签和真实标签的图）；否则为None
    if args.save_umap is True:
        umap_save_path = ['./umap_figure/%s_pred_label.png'%(args.name),'./umap_figure/%s_true_label.png'%(args.name)]
    else:
        umap_save_path = [None, None]

        # 调用utils.load_data加载数据，输入路径为'./Data/AnnData/文件名'，返回：
        # adata（AnnData对象，单细胞数据标准格式）、rawData（原始基因表达数据）、
        # dataset（预处理后的数据，可能经过高变基因筛选）、adj和r_adj（细胞间邻接矩阵）
    adata, rawData, dataset, adj, r_adj = utils.load_data('./Data/AnnData/{}'.format(args.name),args=args)
    # 如果细胞类型已知（celltype="known"），从adata.obs['celltype']获取真实标签列表；否则为None
    if args.celltype == "known":  
        celltype = adata.obs['celltype'].tolist()
    else:
        celltype = None
        # 当细胞数量小于max_num_cell时，直接使用全部数据训练
    if adata.shape[0] < args.max_num_cell:
        size_factor = adata.obs['size_factors'].values  # 获取细胞大小因子（用于标准化）
        Zscore_data = preprocessing.scale(dataset)  # 对数据进行Z-score标准化（均值0，方差1）
        
        args.n_input = dataset.shape[1]  # 设置模型输入维度为基因数量（数据集的列数）,行是细胞,列是基因
        print(args)             # 打印参数配置
        # 初始化注意力自编码器模型，参数包括中间层维度（256,64,64,256）、输入维度、潜在维度、注意力头数、设备
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads, device=device)
        # 预训练模型：调用train函数，输入初始模型、标准化数据、原始数据、邻接矩阵、大小因子、设备和参数，返回预训练模型
        pretrain_model, _ = train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args)
        # 聚类阶段：调用clustering函数，输入预训练模型、数据、真实标签、邻接矩阵等，返回评估指标、预测标签、模型等
        metric, pred_label, _, model, _ = clustering(pretrain_model, Zscore_data, rawData, celltype,
                                                  adj, r_adj, size_factor, device, args)
        asw = metric[0]     # 从metric中提取轮廓系数（ASW，聚类紧凑性指标）
        db  = metric[1]     # 从metric中提取戴维斯-布尔丁指数（DB，聚类分离度指标）
        # 如果有真实标签，计算并打印NMI（标准化互信息）和ARI（调整兰德指数）
        if celltype is not None:
            nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
            ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
            print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))  # 打印评估指标
            # 如果没有真实标签，只打印ASW和DB
        else:
            print("Final ASW %.3f, DB %.3f"% (asw, db))
            # 将标准化数据转换为PyTorch张量，并移动到指定设备（CPU/GPU）
        data = torch.Tensor(Zscore_data).to(device)
        # 处理邻接矩阵：如果是scipy稀疏矩阵，转换为PyTorch稀疏张量并移动到设备；否则转换为张量并放在CPU
        if type(adj) == scipy.sparse._csr.csr_matrix:
            adj = utils.sparse_mx_to_torch_sparse_tnsor(adj).to(device)
        else:
            adj = torch.Tensor(adj).cpu()
            # 禁用梯度计算（节省内存，仅用于推理）
        with torch.no_grad():
            # 用训练好的模型生成潜在向量z（其他返回值为中间变量，用_忽略）
            z, _, _, _, _, _, _ = model(data,adj)
            # 如果需要保存UMAP，调用utils.umap_visual绘制预测标签的UMAP图
            if args.save_umap is True:
                utils.umap_visual(z.detach().cpu().numpy(),   # 潜在向量转换为NumPy数组（从GPU移到CPU）
                                  label = pred_label,     # 预测标签
                                  title='predicted label', # 图标题
                                  save_path = umap_save_path[0],  # 保存路径
                                  asw_used=True) # 使用ASW相关配置
                # 如果细胞类型已知，绘制真实标签的UMAP图
                if args.celltype == "known":  
                    utils.umap_visual(z.detach().cpu().numpy(), 
                                      label = celltype, 
                                      title='true label', 
                                      save_path = umap_save_path[1])
                    # 如果需要保存嵌入向量，将z保存到./embedding/文件名.csv
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
            # 如果需要保存预测标签，将pred_label保存到./pred_label/文件名.csv
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name),pred_label)
            # 如果需要保存模型参数，将模型状态字典保存到./model_save/文件名.pkl
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
        
        # output predicted labels
        # np.savetxt('./results/%s_predicted_label.csv'%(args.name),pred_label)
    # 当细胞数量超过max_num_cell时，使用下采样策略
    #down-sampling input
    else:
        # 对adata进行下采样，保留max_num_cell个细胞，返回新的adata（new_adata）
        new_adata = utils.random_downsimpling(adata, args.max_num_cell)
        # 基于下采样数据构建新的邻接矩阵new_adj和new_r_adj（使用指定的连接性方法和近邻数）
        new_adj, new_r_adj = utils.adata_knn(new_adata, method = args.connectivity_methods, 
                                             knn=args.knn, n_neighbors=args.n_neighbors)
        # 尝试将下采样数据转换为数组并标准化（处理稀疏矩阵和稠密矩阵两种情况）
        try: 
            new_Zscore_data = preprocessing.scale(new_adata.X.toarray())  # 若X是稀疏矩阵，转为数组后标准化
            new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X.toarray()   # 提取下采样的原始高变基因数据
        except:
            new_Zscore_data = preprocessing.scale(new_adata.X)  # 若X是稠密矩阵，直接标准化
            new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X  # 提取原始高变基因数据

        # 获取下采样数据的细胞大小因子
        size_factor = new_adata.obs['size_factors'].values
        # 对全量数据进行标准化（同样处理稀疏和稠密情况）
        try: 
            Zscore_data = preprocessing.scale(dataset.toarray())
            
        except:
            Zscore_data = preprocessing.scale(dataset)

        # 获取下采样数据的真实细胞类型标签
        new_celltype = new_adata.obs['celltype']
        args.n_input = dataset.shape[1]  # 设置模型输入维度为基因数量
        print(args)   # 打印参数配置
        # 初始化注意力自编码器模型（同前）
        init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads,device=device)
        # 用下采样数据预训练模型，返回预训练模型和训练时间
        pretrain_model, train_elapsed_time  = train(init_model, new_Zscore_data, new_rawData,
                                                    new_adj, new_r_adj, size_factor, device, args)
        # 用下采样数据进行聚类阶段训练，返回聚类相关结果（重点获取cluster_layer和训练好的模型）
        _, _, cluster_layer, model, _ = clustering(pretrain_model, new_Zscore_data, new_rawData, 
                                                   new_celltype, new_adj, new_r_adj, size_factor, device, args)
        # 创建一个在CPU上的模型副本（用于全量数据推理，避免GPU内存不足）
        copy_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads, device=torch.device('cpu'))
        copy_model.load_state_dict(model.state_dict())  # 加载训练好的模型参数
        # 将全量标准化数据转换为CPU上的张量（因副本模型在CPU）
        data = torch.Tensor(Zscore_data).cpu()
        # 处理全量数据的邻接矩阵（同前，转换为合适的张量类型）
        if type(adj) == scipy.sparse._csr.csr_matrix:
            adj = utils.sparse_mx_to_torch_sparse_tnsor(adj).to(device)
        else:
            adj = torch.Tensor(adj).cpu()
            # 禁用梯度计算，用副本模型处理全量数据
        with torch.no_grad():
            # 生成全量数据的潜在向量z
            z, _, _, _, _, _, _ = model(data,adj)
            # 计算损失函数，得到预测概率p（用于生成标签）
            _, p = loss_func(z, cluster_layer.cpu())
            # 将概率转换为预测标签（通过距离计算）
            pred_label = utils.dist_2_label(p)
            # 如果有真实标签，计算并打印所有评估指标
            if args.celltype == "known":  
                asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                db = np.round(davies_bouldin_score(z.detach().cpu().numpy(), pred_label), 3)
                nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
                ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
                print("Final ASW %.3f, DB %.3f, ARI %.3f, NMI %.3f"% (asw, db, ari, nmi))
            else:  # 无真实标签时仅打印ASW和DB
                asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                db = np.round(davies_bouldin_score(z.detach().cpu().numpy(), pred_label), 3)
                print("Final ASW %.3f, DB %.3f"% (asw,db))
                # 如果需要保存UMAP，绘制并保存预测标签和真实标签的图（同前）
            if args.save_umap is True:
                utils.umap_visual(z.detach().cpu().numpy(), 
                                  label = pred_label, 
                                  title='predicted label', 
                                  save_path = umap_save_path[0],
                                  asw_used=True)
                if args.celltype == "known":  
                    utils.umap_visual(z.detach().cpu().numpy(), 
                                      label = celltype, 
                                      title='true label', 
                                      save_path = umap_save_path[1])
                    # 保存嵌入向量（全量数据的z）
        if args.save_embedding is True:
            np.savetxt('./embedding/%s.csv'%(args.name), z.detach().cpu().numpy())
            # 保存预测标签（全量数据的pred_label）
        if args.save_pred_label is True:
            np.savetxt('./pred_label/%s.csv'%(args.name), pred_label)
            # 保存模型参数
        if args.save_model_para is True:
            torch.save(model.state_dict(), './model_save/%s.pkl'%(args.name))
