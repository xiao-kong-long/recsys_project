"""Item2Vec - 将电影编码为 embedding 向量

基于用户的观影历史序列，使用 Word2Vec 学习电影向量表示。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None
    print("警告: 未安装 gensim，请运行: pip install gensim")


def prepare_user_sequences(
    ratings_df: pd.DataFrame,
    min_items: int = 2,
    sort_by: str = "timestamp"
) -> List[List[int]]:
    """准备用户观影序列数据

    将每个用户的评分历史按时间排序，转换为序列列表。
    每个序列代表一个用户的观影路径。

    Args:
        ratings_df: 评分数据 DataFrame (user_id, item_id, rating, timestamp)
        min_items: 用户至少需要有 min_items 条评分记录才会被包含
        sort_by: 排序字段，默认按 timestamp 排序

    Returns:
        用户观影序列列表，每个序列是一个 movie_id 列表
    """
    sequences = []

    for user_id, group in ratings_df.groupby("user_id"):
        if len(group) < min_items:
            continue

        # 按时间排序
        sorted_items = group.sort_values(sort_by)["item_id"].tolist()
        sequences.append(sorted_items)

    return sequences


def train_item2vec(
    sequences: List[List[int]],
    embedding_dim: int = 128,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
    epochs: int = 10,
    **kwargs
) -> Optional[Word2Vec]:
    """训练 Item2Vec 模型

    Args:
        sequences: 用户观影序列列表
        embedding_dim: embedding 维度
        window: 上下文窗口大小
        min_count: 最小出现次数
        sg: 1=Skip-gram, 0=CBOW
        epochs: 训练轮数
        **kwargs: 其他 Word2Vec 参数

    Returns:
        训练好的 Word2Vec 模型
    """
    if Word2Vec is None:
        raise ImportError("请先安装 gensim: pip install gensim")

    # 将 item_id 转换为字符串（Word2Vec 需要字符串类型的 key）
    str_sequences = [[str(item_id) for item_id in seq] for seq in sequences]

    model = Word2Vec(
        sentences=str_sequences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        **kwargs
    )

    return model


class Item2Vec:
    """Item2Vec 模型封装类

    用法:
        # 训练
        model = Item2Vec(embedding_dim=128)
        model.fit(ratings_df)

        # 获取 embedding
        embedding = model.get_item_embedding(item_id=1)

        # 获取相似电影
        similar = model.similar_items(item_id=1, top_k=10)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        window: int = 5,
        min_count: int = 1,
        sg: int = 1,
        epochs: int = 10
    ):
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model: Optional[Word2Vec] = None

    def fit(self, ratings_df: pd.DataFrame, **kwargs) -> "Item2Vec":
        """训练模型

        Args:
            ratings_df: 评分数据 DataFrame
            **kwargs: 额外的训练参数

        Returns:
            self
        """
        sequences = prepare_user_sequences(ratings_df)
        self.model = train_item2vec(
            sequences=sequences,
            embedding_dim=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            **kwargs
        )
        return self

    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """获取单个电影的 embedding 向量

        Args:
            item_id: 电影 ID

        Returns:
            embedding 向量，如果电影不存在返回 None
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        try:
            return self.model.wv[str(item_id)]
        except KeyError:
            return None

    def get_embeddings_matrix(self, num_items: int) -> np.ndarray:
        """获取所有电影的 embedding 矩阵

        Args:
            num_items: 电影总数

        Returns:
            shape: (num_items, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        embeddings = np.zeros((num_items, self.embedding_dim))

        for i in range(1, num_items + 1):
            emb = self.get_item_embedding(i)
            if emb is not None:
                embeddings[i - 1] = emb

        return embeddings

    def similar_items(self, item_id: int, top_k: int = 10) -> List[tuple]:
        """获取与指定电影最相似的电影

        Args:
            item_id: 电影 ID
            top_k: 返回前 k 个相似电影

        Returns:
            [(movie_id, similarity), ...] 按相似度降序排列
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        try:
            similar = self.model.wv.most_similar(str(item_id), topn=top_k)
            return [(int(item_id), similarity) for item_id, similarity in similar]
        except KeyError:
            return []

    def save(self, filepath: str) -> None:
        """保存模型

        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        self.model.save(filepath)

    @classmethod
    def load(cls, filepath: str, **init_kwargs) -> "Item2Vec":
        """加载模型

        Args:
            filepath: 模型文件路径
            **init_kwargs: 初始化参数

        Returns:
            Item2Vec 实例
        """
        if Word2Vec is None:
            raise ImportError("请先安装 gensim: pip install gensim")

        instance = cls(**init_kwargs)
        instance.model = Word2Vec.load(filepath)
        # 从加载的模型中获取实际的向量维度
        instance.embedding_dim = instance.model.wv.vector_size
        return instance


if __name__ == "__main__":
    from dataset import load_movielens_100k

    # 加载数据
    data = load_movielens_100k()

    print("=" * 50)
    print("Item2Vec 训练示例")
    print("=" * 50)
    print(f"用户数: {len(data['users'])}")
    print(f"电影数: {len(data['items'])}")
    print(f"评分数: {len(data['ratings'])}")

    # 准备用户序列
    sequences = prepare_user_sequences(data["ratings"])
    print(f"\n有效用户序列数: {len(sequences)}")

    # 训练模型
    print("\n开始训练 Item2Vec 模型...")
    model = Item2Vec(embedding_dim=64, epochs=5)
    model.fit(data["ratings"])
    print("训练完成!")

    # 获取 embedding
    item_id = 1
    embedding = model.get_item_embedding(item_id)
    print(f"\n电影 {item_id} 的 embedding 维度: {embedding.shape}")

    # 获取相似电影
    similar = model.similar_items(item_id, top_k=5)
    print(f"\n与电影 {item_id} 最相似的 5 部电影:")
    for sim_item_id, similarity in similar:
        movie_name = data["items"][data["items"]["movie_id"] == sim_item_id]["title"].values[0]
        print(f"  {movie_name}: {similarity:.4f}")

    # 保存模型
    model_path = "item2vec_model.bin"
    model.save(model_path)
    print(f"\n模型已保存到: {model_path}")
    print("运行评估: python evaluate.py")