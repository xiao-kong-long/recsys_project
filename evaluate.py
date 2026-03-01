"""Item2Vec Embedding 评估工具

提供多种方法评估 embedding 质量：
1. 直观检查相似电影
2. 相似度分布分析
3. t-SNE 降维可视化
4. 聚类评估
5. 推荐任务评估
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
except ImportError as e:
    print(f"警告: 请安装必要的库: pip install scikit-learn matplotlib")


def check_similar_movies(
    model,
    items_df: pd.DataFrame,
    item_ids: List[int],
    top_k: int = 10,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """直观检查相似电影

    Args:
        model: 训练好的 Item2Vec 模型
        items_df: 电影数据 DataFrame
        item_ids: 要检查的电影 ID 列表
        top_k: 返回前 k 个相似电影
        output_file: 结果保存文件路径

    Returns:
        包含相似电影结果的 DataFrame
    """
    results = []

    for item_id in item_ids:
        movie_info = items_df[items_df["movie_id"] == item_id]
        if len(movie_info) == 0:
            continue

        title = movie_info["title"].values[0]

        similar = model.similar_items(item_id, top_k=top_k)

        for sim_item_id, similarity in similar:
            sim_movie_info = items_df[items_df["movie_id"] == sim_item_id]
            if len(sim_movie_info) > 0:
                sim_title = sim_movie_info["title"].values[0]

                genres = get_item_genres(movie_info.iloc[0])
                sim_genres = get_item_genres(sim_movie_info.iloc[0])

                results.append({
                    "item_id": item_id,
                    "title": title,
                    "genres": ", ".join(genres),
                    "similar_id": sim_item_id,
                    "similar_title": sim_title,
                    "similar_genres": ", ".join(sim_genres),
                    "similarity": round(similarity, 4)
                })

    df = pd.DataFrame(results)

    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"相似电影结果已保存到: {output_file}")

    return df


def analyze_similarity_distribution(
    embeddings: np.ndarray,
    output_dir: str = "results"
) -> Dict[str, float]:
    """分析相似度分布

    Args:
        embeddings: embedding 矩阵, shape (n_items, embedding_dim)
        output_dir: 图片保存目录

    Returns:
        统计指标字典
    """
    os.makedirs(output_dir, exist_ok=True)

    print("计算相似度矩阵...")
    sim_matrix = cosine_similarity(embeddings)

    n = sim_matrix.shape[0]
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]

    stats = {
        "mean": float(np.mean(upper_tri)),
        "std": float(np.std(upper_tri)),
        "min": float(np.min(upper_tri)),
        "max": float(np.max(upper_tri)),
        "median": float(np.median(upper_tri)),
        "q25": float(np.percentile(upper_tri, 25)),
        "q75": float(np.percentile(upper_tri, 75))
    }

    # 保存统计结果
    stats_file = os.path.join(output_dir, "similarity_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("相似度分布统计\n")
        f.write("=" * 40 + "\n")
        for k, v in stats.items():
            f.write(f"{k:10s}: {v:.4f}\n")

    # 绘制直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(upper_tri, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Similarity Distribution")
    axes[0].grid(True, alpha=0.3)

    for stat_name, stat_value, color in [("Mean", stats["mean"], "red"),
                                          ("Median", stats["median"], "blue")]:
        axes[0].axvline(stat_value, color=color, linestyle='--',
                       label=f"{stat_name}: {stat_value:.3f}", linewidth=2)
    axes[0].legend()

    # 箱线图
    axes[1].boxplot(upper_tri, vert=False)
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_title("Similarity Boxplot")
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    hist_file = os.path.join(output_dir, "similarity_distribution.png")
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"相似度分布图已保存到: {hist_file}")

    return stats


def visualize_embeddings_tsne(
    embeddings: np.ndarray,
    items_df: pd.DataFrame,
    genre_names: List[str],
    output_dir: str = "results",
    n_samples: int = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 t-SNE 降维可视化 embedding

    Args:
        embeddings: embedding 矩阵
        items_df: 电影数据 DataFrame
        genre_names: 类型名称列表
        output_dir: 图片保存目录
        n_samples: 采样数量，None 表示使用全部
        random_state: 随机种子

    Returns:
        (embeddings_2d, labels) 二维 embedding 和对应标签
    """
    os.makedirs(output_dir, exist_ok=True)

    # 采样（可选）
    if n_samples is not None and n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        items_df = items_df.iloc[indices].reset_index(drop=True)

    # 获取主类型
    labels = []
    label_names = []
    for _, row in items_df.iterrows():
        for genre in genre_names:
            if row.get(genre, 0) == 1:
                labels.append(genre_names.index(genre))
                label_names.append(genre)
                break
        else:
            labels.append(-1)
            label_names.append("unknown")

    # t-SNE 降维
    print(f"执行 t-SNE 降维 (n_samples={len(embeddings)})...")
    tsne = TSNE(n_components=2, random_state=random_state,
                perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制散点图
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 颜色映射
    unique_labels = sorted(set(l for l in labels if l != -1))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # 左图：按类型着色
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[i]], label=genre_names[label], alpha=0.6, s=20)

    # 未知类型
    if -1 in labels:
        mask = np.array(labels) == -1
        axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c='gray', label='unknown', alpha=0.3, s=10)

    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].set_title("t-SNE Visualization by Genre")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 右图：标记电影标题
    popular_indices = np.random.choice(len(embeddings_2d),
                                       min(20, len(embeddings_2d)), replace=False)
    for idx in popular_indices:
        axes[1].scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                       c='red', s=100, edgecolors='black', linewidths=1)
        title = items_df.iloc[idx]["title"]
        short_title = title[:30] + "..." if len(title) > 30 else title
        axes[1].annotate(short_title, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                        fontsize=6, xytext=(5, 5), textcoords='offset points')

    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].set_title("t-SNE Visualization (Sampled Movies)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_file = os.path.join(output_dir, "tsne_visualization.png")
    plt.savefig(tsne_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"t-SNE 可视化图已保存到: {tsne_file}")

    return embeddings_2d, np.array(labels)


def evaluate_clustering(
    embeddings: np.ndarray,
    items_df: pd.DataFrame,
    genre_names: List[str],
    output_dir: str = "results"
) -> Dict[str, float]:
    """评估聚类质量

    Args:
        embeddings: embedding 矩阵
        items_df: 电影数据 DataFrame
        genre_names: 类型名称列表
        output_dir: 结果保存目录

    Returns:
        评估指标字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取真实标签（主类型）
    true_labels = []
    for _, row in items_df.iterrows():
        for i, genre in enumerate(genre_names):
            if row.get(genre, 0) == 1:
                true_labels.append(i)
                break
        else:
            true_labels.append(0)

    true_labels = np.array(true_labels)

    # 不同聚类数量的评估
    n_clusters_options = [5, 10, 15, 19]
    results = []

    for n_clusters in n_clusters_options:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, cluster_labels)

        results.append({
            "n_clusters": n_clusters,
            "silhouette_score": silhouette
        })

    df = pd.DataFrame(results)
    result_file = os.path.join(output_dir, "clustering_evaluation.csv")
    df.to_csv(result_file, index=False)
    print(f"聚类评估结果已保存到: {result_file}")

    # 使用19个类型的详细评估
    kmeans = KMeans(n_clusters=19, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, cluster_labels)

    # 聚类可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(19), [k for k in range(19)],
           alpha=0.7, label='True Genre Distribution')
    ax.set_xlabel("Cluster/Genre Index")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Evaluation (Silhouette Score: {silhouette:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    cluster_file = os.path.join(output_dir, "clustering_evaluation.png")
    plt.savefig(cluster_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"聚类评估图已保存到: {cluster_file}")

    return {"silhouette_score": silhouette, "n_clusters": 19}


def get_item_genres(item_row: pd.Series, genre_names: List[str] = None) -> List[str]:
    """获取电影类型列表"""
    if genre_names is None:
        from dataset import get_genre_names
        genre_names = get_genre_names()

    return [genre for genre in genre_names if item_row.get(genre, 0) == 1]


def evaluate_recommendation(
    model,
    test_ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    train_ratings_df: pd.DataFrame = None,
    top_k: int = 10,
    output_dir: str = "results"
) -> Dict[str, float]:
    """评估推荐任务性能

    Args:
        model: 训练好的模型 (Item2Vec 或 TwoTower)
        test_ratings_df: 测试集评分数据
        items_df: 电影数据
        train_ratings_df: 训练集评分数据（用于获取用户历史）
        top_k: 推荐列表长度
        output_dir: 结果保存目录

    Returns:
        评估指标字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 如果没有提供训练集，使用测试集作为用户历史
    user_history = train_ratings_df if train_ratings_df is not None else test_ratings_df

    # 按用户组织测试集
    test_items_by_user = defaultdict(list)
    for user_id, group in test_ratings_df.groupby("user_id"):
        test_items_by_user[user_id] = set(group["item_id"].tolist())

    # 计算评估指标
    precision_scores = []

    for user_id, test_items in test_items_by_user.items():
        if not test_items:
            continue

        # 获取用户历史评分的物品
        user_items = user_history[user_history["user_id"] == user_id]["item_id"].tolist()

        for test_item in test_items:
            # 根据模型类型获取推荐
            if hasattr(model, 'similar_items'):
                # Item2Vec: 获取与测试项目相似的电影
                similar = model.similar_items(test_item, top_k=top_k)
                similar_ids = [sid for sid, _ in similar]
            elif hasattr(model, 'recommend'):
                # TwoTower: 直接为用户推荐
                recommendations = model.recommend(user_id, top_k=top_k, exclude_rated=False)
                similar_ids = [sid for sid, _ in recommendations]
            else:
                continue

            # 计算命中数：推荐列表中，用户在测试集也评分过的电影
            hits = len(set(similar_ids) & (set(test_items) - {test_item}))
            precision = hits / top_k if top_k > 0 else 0

            precision_scores.append(precision)

    metrics = {
        "precision@k": np.mean(precision_scores) if precision_scores else 0,
        "top_k": top_k,
        "num_users_evaluated": len(test_items_by_user)
    }

    result_file = os.path.join(output_dir, "recommendation_evaluation.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("推荐任务评估结果\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:20s}: {v:.4f}\n")

    print(f"推荐评估结果已保存到: {result_file}")

    return metrics


def full_evaluation(
    model,
    data: Dict[str, pd.DataFrame],
    output_dir: str = "results"
) -> Dict[str, any]:
    """完整评估流程

    Args:
        model: 训练好的模型 (Item2Vec 或 TwoTower)
        data: 包含 users, items, ratings 的字典，
              或者包含 users, items, train_ratings, test_ratings 的字典
        output_dir: 结果保存目录

    Returns:
        所有评估结果
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("开始完整评估流程")
    print("=" * 50)

    results = {}

    # 1. 直观检查相似电影
    print("\n[1/5] 直观检查相似电影...")
    sample_movies = data["items"]["movie_id"].head(10).tolist()
    similar_df = check_similar_movies(
        model, data["items"], sample_movies,
        top_k=5,
        output_file=os.path.join(output_dir, "similar_movies.csv")
    )
    print("\n相似电影示例:")
    print(similar_df.head(10).to_string(index=False))
    results["similar_movies"] = similar_df

    # 2. 相似度分布分析
    print("\n[2/5] 相似度分布分析...")
    embeddings = model.get_embeddings_matrix(len(data["items"]))
    stats = analyze_similarity_distribution(embeddings, output_dir)
    results["similarity_stats"] = stats
    print(f"相似度均值: {stats['mean']:.4f}")

    # 3. t-SNE 可视化
    print("\n[3/5] t-SNE 可视化...")
    from dataset import get_genre_names
    genre_names = get_genre_names()
    tsne_result = visualize_embeddings_tsne(
        embeddings, data["items"], genre_names,
        output_dir, n_samples=500
    )
    results["tsne"] = tsne_result

    # 4. 聚类评估
    print("\n[4/5] 聚类评估...")
    cluster_results = evaluate_clustering(embeddings, data["items"], genre_names, output_dir)
    results["clustering"] = cluster_results
    print(f"轮廓系数: {cluster_results['silhouette_score']:.4f}")

    # 5. 推荐任务评估
    print("\n[5/5] 推荐任务评估...")
    # 判断数据是否包含训练集和测试集
    if "test_ratings" in data and "train_ratings" in data:
        rec_results = evaluate_recommendation(
            model, data["test_ratings"], data["items"],
            train_ratings_df=data["train_ratings"], output_dir=output_dir
        )
    else:
        rec_results = evaluate_recommendation(model, data["ratings"], data["items"], output_dir=output_dir)
    results["recommendation"] = rec_results
    print(f"Precision@10: {rec_results['precision@k']:.4f}")

    print("\n" + "=" * 50)
    print(f"评估完成！结果已保存到: {output_dir}/")
    print("=" * 50)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估推荐模型")
    parser.add_argument("--model", type=str, default="item2vec", choices=["item2vec", "twotower"], help="模型类型")
    parser.add_argument("--model-path", type=str, default=None, help="模型路径（默认使用 item2vec_model.bin 或 models/twotower_model.pt）")
    parser.add_argument("--data-path", type=str, default="data/ml-100k", help="数据集路径")
    parser.add_argument("--split", type=str, default="ua", choices=["ua", "ub"], help="数据划分方式")
    parser.add_argument("--output-dir", type=str, default="results", help="结果保存目录")

    args = parser.parse_args()

    # 加载数据（使用训练/测试集划分）
    print(f"加载数据...")
    from dataset import load_movielens_train_test
    data = load_movielens_train_test(args.data_path, split=args.split)

    print(f"训练集评分数: {len(data['train_ratings'])}")
    print(f"测试集评分数: {len(data['test_ratings'])}")

    # 加载模型
    print(f"加载模型...")
    if args.model == "item2vec":
        from item2vec import Item2Vec
        model_path = args.model_path or "item2vec_model.bin"
        model = Item2Vec.load(model_path)
    else:  # twotower
        from twotower import TwoTowerModel
        model_path = args.model_path or "models/twotower_model.pt"
        model = TwoTowerModel.load(model_path)

    results = full_evaluation(model, data, output_dir=args.output_dir)