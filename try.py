import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

# 1. 生成一些二维随机点
np.random.seed(0)
n_samples = 300
blob_centers = np.array([[2, 2], [8, 3], [3, 6], [-1, -1]])
X, y_true = make_blobs(n_samples=n_samples, centers=blob_centers,
                       cluster_std=0.70, random_state=0)

# K-means 
n_clusters = 2  
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X)

#  DBSCAN 
eps = 0.3  # 邻域半径
min_samples = 5  # 最小样本数
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
y_dbscan = dbscan.fit_predict(X)

# 4. 创建子图进行对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='tab10')
axes[0].scatter(blob_centers[:, 0], blob_centers[:, 1], c='red', s=200, 
                alpha=0.75, marker='x', linewidth=3, label='True Centers')
axes[0].set_title('initial data')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].legend()
axes[0].grid(True)

# K-means 
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                alpha=0.75, marker='x', linewidth=3, label='K-means Centers')
axes[1].set_title(f'K-means  (k={n_clusters})')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()
axes[1].grid(True)

# DBSCAN 
unique_labels = set(y_dbscan)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声点用黑色表示
        col = 'black'
        marker = 'x'
        alpha = 0.6
        label = 'Noise'
    else:
        marker = 'o'
        alpha = 0.8
        label = f'Cluster {k}'
    
    class_member_mask = (y_dbscan == k)
    xy = X[class_member_mask]
    axes[2].scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=alpha, 
                    marker=marker, label=label if k == -1 or k == 0 else "")

axes[2].set_title(f'DBSCAN (eps={eps}, min_samples={min_samples})')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


print("聚类结果对比:")
print(f"K-means 聚类数: {len(set(y_kmeans))}")
print(f"DBSCAN 聚类数: {len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)}")
print(f"DBSCAN 噪声点数: {list(y_dbscan).count(-1)}")

# # 6. 尝试不同的DBSCAN参数
# print("\n尝试不同的DBSCAN参数:")
# eps_values = [0.5, 1.0, 1.5]
# min_samples_values = [3, 5, 10]

# fig, axes = plt.subplots(len(eps_values), len(min_samples_values), 
#                          figsize=(15, 12))

# for i, eps in enumerate(eps_values):
#     for j, min_samples in enumerate(min_samples_values):
#         dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
#         y_temp = dbscan_temp.fit_predict(X)
        
#         unique_labels = set(y_temp)
#         colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
#         for k, col in zip(unique_labels, colors):
#             if k == -1:
#                 col = 'black'
#                 marker = 'x'
#                 alpha = 0.6
#             else:
#                 marker = 'o'
#                 alpha = 0.8
            
#             class_member_mask = (y_temp == k)
#             xy = X[class_member_mask]
#             axes[i, j].scatter(xy[:, 0], xy[:, 1], c=[col], s=30, 
#                               alpha=alpha, marker=marker)
        
#         n_clusters_temp = len(set(y_temp)) - (1 if -1 in y_temp else 0)
#         n_noise = list(y_temp).count(-1)
#         axes[i, j].set_title(f'eps={eps}, min_samples={min_samples}\n'
#                             f'Clusters: {n_clusters_temp}, Noise: {n_noise}')
#         axes[i, j].grid(True)

# plt.tight_layout()
# plt.show()