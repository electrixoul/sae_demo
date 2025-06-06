import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from models.sae import SparseAutoencoder
import time

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_config(model_path):
    """加载模型和配置"""
    config = {
        "input_size": 784,
        "hidden_size": 1024,
        "k_sparse": 50,
        "num_saes": 5,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, config, device

def extract_feature_vectors(model, ranking_method='l2_norm', top_k=500):
    """
    提取SAE学习到的特征向量并进行排序
    
    参数:
        model: 训练好的SAE模型
        ranking_method: 排序方法 ('l2_norm', 'variance', 'max_activation')
        top_k: 选择前k个特征
    
    返回:
        features: 选择的特征向量
        indices: 特征的原始索引
        scores: 排序得分
    """
    model.eval()
    
    with torch.no_grad():
        # 获取第一个SAE的编码器权重 (1024, 784)
        weights = model.encoders[0].weight.data.cpu().numpy()
        
        # 计算排序得分
        if ranking_method == 'l2_norm':
            # 使用L2范数排序
            scores = np.linalg.norm(weights, axis=1)
            score_name = "L2 Norm"
        elif ranking_method == 'variance':
            # 使用方差排序
            scores = np.var(weights, axis=1)
            score_name = "Variance"
        elif ranking_method == 'max_activation':
            # 使用最大激活值排序
            scores = np.max(np.abs(weights), axis=1)
            score_name = "Max Activation"
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        # 按得分降序排序，选择前top_k个特征
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]
        
        # 提取对应的特征向量
        top_features = weights[top_indices]
        top_scores = scores[top_indices]
        
        print(f"使用 {score_name} 排序方法")
        print(f"选择了前 {top_k} 个特征")
        print(f"得分范围: {top_scores.min():.4f} - {top_scores.max():.4f}")
        
        return top_features, top_indices, top_scores, score_name

def apply_tsne(features, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """
    对特征向量应用t-SNE降维
    
    参数:
        features: 输入特征矩阵
        n_components: 降维后的维数
        perplexity: t-SNE的困惑度参数
        n_iter: 迭代次数
        random_state: 随机种子
    
    返回:
        embedded: 降维后的坐标
    """
    print(f"开始t-SNE降维...")
    print(f"输入特征形状: {features.shape}")
    print(f"参数: perplexity={perplexity}, n_iter={n_iter}")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 应用t-SNE
    start_time = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    
    embedded = tsne.fit_transform(features_scaled)
    end_time = time.time()
    
    print(f"t-SNE完成，耗时: {end_time - start_time:.2f}秒")
    print(f"输出坐标形状: {embedded.shape}")
    
    return embedded

def visualize_tsne_basic(embedded, scores, score_name, save_path="tsne_basic.png"):
    """
    基础t-SNE可视化，按得分着色
    """
    plt.figure(figsize=(12, 8))
    
    # 创建散点图，按得分着色
    scatter = plt.scatter(
        embedded[:, 0], 
        embedded[:, 1], 
        c=scores, 
        cmap='viridis', 
        alpha=0.7,
        s=20
    )
    
    plt.colorbar(scatter, label=f'{score_name} Score')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f'SAE Feature t-SNE Visualization (Colored by {score_name})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"基础t-SNE可视化已保存到 {save_path}")
    
    return plt.gcf()

def visualize_tsne_clusters(embedded, scores, score_name, n_clusters=5, save_path="tsne_clusters.png"):
    """
    t-SNE可视化，按得分分组显示
    """
    plt.figure(figsize=(12, 8))
    
    # 按得分分成几个层级
    percentiles = np.linspace(0, 100, n_clusters + 1)
    score_thresholds = np.percentile(scores, percentiles)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        mask = (scores >= score_thresholds[i]) & (scores < score_thresholds[i + 1])
        if i == n_clusters - 1:  # 最后一组包含最大值
            mask = scores >= score_thresholds[i]
        
        plt.scatter(
            embedded[mask, 0], 
            embedded[mask, 1], 
            c=[colors[i]], 
            alpha=0.7,
            s=20,
            label=f'Level {i+1} ({np.sum(mask)} features)'
        )
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f'SAE Feature t-SNE Visualization (Grouped by {score_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"分组t-SNE可视化已保存到 {save_path}")
    
    return plt.gcf()

def visualize_tsne_density(embedded, save_path="tsne_density.png"):
    """
    t-SNE密度图可视化
    """
    plt.figure(figsize=(12, 8))
    
    # 创建密度图
    plt.hist2d(embedded[:, 0], embedded[:, 1], bins=50, cmap='Blues', alpha=0.8)
    plt.colorbar(label='Feature Density')
    
    # 叠加散点图
    plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.3, s=10, c='red')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('SAE Feature t-SNE Density Visualization')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"密度t-SNE可视化已保存到 {save_path}")
    
    return plt.gcf()

def analyze_tsne_clusters(embedded, features, indices, scores, score_name):
    """
    分析t-SNE聚类结果
    """
    print("\n=== t-SNE聚类分析 ===")
    
    # 计算坐标统计
    x_coords = embedded[:, 0]
    y_coords = embedded[:, 1]
    
    print(f"X坐标范围: {x_coords.min():.2f} - {x_coords.max():.2f}")
    print(f"Y坐标范围: {y_coords.min():.2f} - {y_coords.max():.2f}")
    print(f"X坐标标准差: {x_coords.std():.2f}")
    print(f"Y坐标标准差: {y_coords.std():.2f}")
    
    # 分析高得分特征的分布
    high_score_mask = scores > np.percentile(scores, 90)
    high_score_x = x_coords[high_score_mask]
    high_score_y = y_coords[high_score_mask]
    
    print(f"\n高得分特征（前10%）分布:")
    print(f"数量: {np.sum(high_score_mask)}")
    print(f"X坐标平均值: {high_score_x.mean():.2f}")
    print(f"Y坐标平均值: {high_score_y.mean():.2f}")
    
    # 找到最极端的几个点
    center_x, center_y = x_coords.mean(), y_coords.mean()
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # 最远的5个点
    farthest_indices = np.argsort(distances)[-5:]
    print(f"\n最远离中心的5个特征:")
    for i, idx in enumerate(farthest_indices):
        print(f"  特征 {indices[idx]}: 距离={distances[idx]:.2f}, 得分={scores[idx]:.4f}")
    
    # 最接近中心的5个点
    closest_indices = np.argsort(distances)[:5]
    print(f"\n最接近中心的5个特征:")
    for i, idx in enumerate(closest_indices):
        print(f"  特征 {indices[idx]}: 距离={distances[idx]:.2f}, 得分={scores[idx]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="SAE特征t-SNE可视化")
    parser.add_argument('--model', type=str, default="mnist_sae_models/mnist_sae_epoch_4.pth",
                      help='模型路径')
    parser.add_argument('--top_k', type=int, default=500,
                      help='选择前k个特征进行t-SNE')
    parser.add_argument('--ranking_method', type=str, choices=['l2_norm', 'variance', 'max_activation'], 
                      default='l2_norm', help='特征排序方法')
    parser.add_argument('--perplexity', type=int, default=30,
                      help='t-SNE困惑度参数')
    parser.add_argument('--n_iter', type=int, default=1000,
                      help='t-SNE迭代次数')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs("visualizations", exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model, config, device = load_model_and_config(args.model)
    
    # 提取和排序特征向量
    features, indices, scores, score_name = extract_feature_vectors(
        model, 
        ranking_method=args.ranking_method, 
        top_k=args.top_k
    )
    
    # 应用t-SNE
    embedded = apply_tsne(
        features, 
        perplexity=args.perplexity, 
        n_iter=args.n_iter
    )
    
    # 生成不同类型的可视化
    print("\n生成可视化图像...")
    
    # 基础可视化（按得分着色）
    visualize_tsne_basic(
        embedded, scores, score_name, 
        save_path=f"visualizations/tsne_basic_{args.ranking_method}.png"
    )
    
    # 分组可视化
    visualize_tsne_clusters(
        embedded, scores, score_name, 
        save_path=f"visualizations/tsne_clusters_{args.ranking_method}.png"
    )
    
    # 密度可视化
    visualize_tsne_density(
        embedded, 
        save_path=f"visualizations/tsne_density_{args.ranking_method}.png"
    )
    
    # 分析聚类结果
    analyze_tsne_clusters(embedded, features, indices, scores, score_name)
    
    # 保存结果数据
    results = {
        'embedded_coordinates': embedded,
        'feature_indices': indices,
        'feature_scores': scores,
        'score_name': score_name,
        'ranking_method': args.ranking_method,
        'top_k': args.top_k
    }
    
    results_path = f"visualizations/tsne_results_{args.ranking_method}_top{args.top_k}.npz"
    np.savez(results_path, **results)
    print(f"\n结果数据已保存到 {results_path}")
    
    print(f"\n🎉 t-SNE可视化完成！")
    print(f"📁 所有结果都保存在 visualizations/ 目录中")
    print(f"📊 使用了 {score_name} 排序方法")
    print(f"🔍 分析了前 {args.top_k} 个特征")

if __name__ == "__main__":
    main()
