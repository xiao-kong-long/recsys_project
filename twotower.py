"""Two-Tower Model - 双塔推荐模型 (PyTorch 版本)

将用户特征和物品特征分别映射到统一的embedding空间，通过点积计算相似度。
相比 Item2Vec，双塔模型可以利用用户和物品的更多特征信息。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError("请先安装 PyTorch: pip install torch")


class UserTower(nn.Module):
    """用户塔神经网络

    输入: [age_norm, gender, occupation_onehot] (23维)
    输出: user_embedding (embedding_dim维)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 最后一个输出层不加 ReLU
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim) L2 归一化的 embedding
        """
        emb = self.network(x)
        # L2 归一化
        return F.normalize(emb, p=2, dim=-1)


class ItemTower(nn.Module):
    """物品塔神经网络

    输入: [genre_multi_hot] (19维)
    输出: item_embedding (embedding_dim维)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim) L2 归一化的 embedding
        """
        emb = self.network(x)
        return F.normalize(emb, p=2, dim=-1)


class TwoTowerDataset(Dataset):
    """双塔模型训练数据集

    生成 (user_features, pos_item_features, neg_item_features) 三元组
    """

    def __init__(
        self,
        user_features: Dict[int, np.ndarray],
        item_features: Dict[int, np.ndarray],
        ratings_df: pd.DataFrame,
        num_negatives: int = 4,
        min_rating: int = 4
    ):
        self.user_features = user_features
        self.item_features = item_features
        self.num_negatives = num_negatives

        # 构建用户已评分物品集合
        self.user_rated_items = defaultdict(set)
        self.user_items = defaultdict(list)

        # 只使用高评分作为正样本
        high_ratings = ratings_df[ratings_df["rating"] >= min_rating]
        for _, row in high_ratings.iterrows():
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            self.user_items[user_id].append(item_id)

        # 记录所有已评分物品（用于负采样）
        for _, row in ratings_df.iterrows():
            self.user_rated_items[int(row["user_id"])].add(int(row["item_id"]))

        # 准备所有物品ID列表
        self.all_item_ids = list(item_features.keys())
        self.all_item_set = set(self.all_item_ids)

        # 展平样本列表
        self.samples = []
        for user_id, pos_items in self.user_items.items():
            if user_id not in user_features:
                continue
            for pos_item_id in pos_items:
                if pos_item_id in item_features:
                    self.samples.append((user_id, pos_item_id))

    def _sample_negative(self, user_id: int, pos_item_id: int) -> List[int]:
        """采样负样本"""
        rated = self.user_rated_items.get(user_id, set())
        available = [i for i in self.all_item_ids if i not in rated and i != pos_item_id]

        if len(available) == 0:
            return [pos_item_id]  # 如果没有可用负样本，返回正样本（避免错误）

        if self.num_negatives >= len(available):
            return available

        return np.random.choice(available, self.num_negatives, replace=False).tolist()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取一个训练样本

        Returns:
            (user_features, pos_item_features, neg_item_features)
            每个都是 torch.Tensor
        """
        user_id, pos_item_id = self.samples[idx]

        # 获取用户特征
        user_feat = self.user_features[user_id]

        # 获取正样本物品特征
        pos_item_feat = self.item_features[pos_item_id]

        # 采样负样本
        neg_item_ids = self._sample_negative(user_id, pos_item_id)
        neg_item_feats = np.stack([self.item_features[iid] for iid in neg_item_ids])

        return (
            torch.from_numpy(user_feat).float(),
            torch.from_numpy(pos_item_feat).float(),
            torch.from_numpy(neg_item_feats).float()
        )


class TwoTowerModel:
    """双塔模型封装类

    用法:
        # 训练
        model = TwoTowerModel(embedding_dim=64, device='cuda')
        model.fit(users_df, items_df, ratings_df)

        # 获取用户 embedding
        user_emb = model.get_user_embedding(user_id=1)

        # 获取物品 embedding
        item_emb = model.get_item_embedding(item_id=1)

        # 预测用户对物品的偏好
        score = model.score(user_id=1, item_id=1)

        # 推荐
        recommendations = model.recommend(user_id=1, top_k=10)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        user_hidden_dims: List[int] = [128, 64],
        item_hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.001,
        l2_reg: float = 0.01,
        device: str = "auto"
    ):
        """初始化双塔模型

        Args:
            embedding_dim: 最终embedding维度
            user_hidden_dims: 用户塔隐藏层维度列表
            item_hidden_dims: 物品塔隐藏层维度列表
            learning_rate: 学习率
            l2_reg: L2正则化系数
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.embedding_dim = embedding_dim
        self.user_hidden_dims = user_hidden_dims
        self.item_hidden_dims = item_hidden_dims
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 模型相关
        self.user_tower: Optional[UserTower] = None
        self.item_tower: Optional[ItemTower] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # 训练相关
        self.is_fitted = False
        self.num_users = 0
        self.num_items = 0
        self.user_rated_items = defaultdict(set)

        # Embedding 缓存
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.item_embeddings: Dict[int, np.ndarray] = {}
        self.feature_encoders = {}

    def _encode_user_features(
        self,
        users_df: pd.DataFrame
    ) -> Dict[int, np.ndarray]:
        """编码用户特征

        用户特征:
            - age: 归一化到 [0, 1]
            - gender: M=1, F=0
            - occupation: one-hot 编码 (21维)
        """
        from dataset import load_occupations

        feature_dict = {}

        # 获取职业列表
        occupations = load_occupations()
        occ_to_idx = {occ: i for i, occ in enumerate(occupations)}

        # 归一化 age
        ages = users_df["age"].values
        age_min, age_max = ages.min(), ages.max()

        for _, row in users_df.iterrows():
            user_id = int(row["user_id"])

            # age: 归一化
            age_norm = (row["age"] - age_min) / (age_max - age_min + 1e-6)

            # gender: 编码
            gender = 1.0 if row["gender"] == "M" else 0.0

            # occupation: one-hot
            occ_onehot = np.zeros(len(occupations))
            occ = row["occupation"]
            if occ in occ_to_idx:
                occ_onehot[occ_to_idx[occ]] = 1.0

            # 拼接所有特征
            feature = np.concatenate([
                [age_norm, gender],
                occ_onehot
            ])

            feature_dict[user_id] = feature

        # 保存归一化参数
        self.feature_encoders["age_min"] = age_min
        self.feature_encoders["age_max"] = age_max
        self.feature_encoders["occupations"] = occupations

        return feature_dict

    def _encode_item_features(
        self,
        items_df: pd.DataFrame
    ) -> Dict[int, np.ndarray]:
        """编码物品特征

        物品特征:
            - genre: 19维 multi-hot
        """
        from dataset import get_genre_names

        feature_dict = {}
        genre_names = get_genre_names()

        for _, row in items_df.iterrows():
            item_id = int(row["movie_id"])

            # genre: multi-hot (19维)
            genre = row[genre_names].values.astype(float)

            feature_dict[item_id] = genre

        self.feature_encoders["genre_names"] = genre_names

        return feature_dict

    def _build_model(
        self,
        user_feature_dim: int,
        item_feature_dim: int
    ) -> None:
        """构建模型"""
        self.user_tower = UserTower(
            user_feature_dim, self.user_hidden_dims, self.embedding_dim
        ).to(self.device)

        self.item_tower = ItemTower(
            item_feature_dim, self.item_hidden_dims, self.embedding_dim
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.user_tower.parameters()) + list(self.item_tower.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )

    def _bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor
    ) -> torch.Tensor:
        """计算 BPR 损失

        Args:
            user_emb: (batch_size, embedding_dim)
            pos_item_emb: (batch_size, embedding_dim)
            neg_item_emb: (batch_size * num_negatives, embedding_dim)

        Returns:
            标量损失
        """
        # 正样本分数
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # (batch_size,)

        # 负样本分数
        neg_item_emb = neg_item_emb.view(-1, self.num_negatives, self.embedding_dim)
        neg_scores = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=2)  # (batch_size, num_negatives)

        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).sum()

        return loss / len(user_emb)

    def fit(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 256,
        num_negatives: int = 4,
        min_rating: int = 4,
        verbose: bool = True
    ) -> "TwoTowerModel":
        """训练双塔模型

        Args:
            users_df: 用户数据
            items_df: 物品数据
            ratings_df: 评分数据
            epochs: 训练轮数
            batch_size: 批大小
            num_negatives: 每个正样本采样的负样本数
            min_rating: 作为正样本的最低评分
            verbose: 是否打印训练信息

        Returns:
            self
        """
        self.num_negatives = num_negatives

        if verbose:
            print("=" * 50)
            print("Two-Tower 模型训练 (PyTorch)")
            print("=" * 50)
            print(f"设备: {self.device}")

        # 编码特征
        if verbose:
            print("编码用户特征...")
        user_features = self._encode_user_features(users_df)

        if verbose:
            print("编码物品特征...")
        item_features = self._encode_item_features(items_df)

        # 记录统计信息
        self.num_users = len(users_df)
        self.num_items = len(items_df)

        # 记录用户已评分物品
        for user_id, item_id in ratings_df[["user_id", "item_id"]].values:
            self.user_rated_items[int(user_id)].add(int(item_id))

        # 构建模型
        user_feature_dim = next(iter(user_features.values())).shape[0]
        item_feature_dim = next(iter(item_features.values())).shape[0]

        self._build_model(user_feature_dim, item_feature_dim)

        if verbose:
            print(f"用户特征维度: {user_feature_dim}")
            print(f"物品特征维度: {item_feature_dim}")
            print(f"用户塔架构: {user_feature_dim} -> {self.user_hidden_dims} -> {self.embedding_dim}")
            print(f"物品塔架构: {item_feature_dim} -> {self.item_hidden_dims} -> {self.embedding_dim}")

        # 创建数据集和数据加载器
        dataset = TwoTowerDataset(
            user_features,
            item_features,
            ratings_df,
            num_negatives=num_negatives,
            min_rating=min_rating
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        if verbose:
            print(f"正样本数: {len(dataset)}")
            print(f"\n开始训练...")

        # 训练循环
        for epoch in range(epochs):
            self.user_tower.train()
            self.item_tower.train()

            total_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                user_feat, pos_item_feat, neg_item_feat = batch
                user_feat = user_feat.to(self.device)
                pos_item_feat = pos_item_feat.to(self.device)
                neg_item_feat = neg_item_feat.to(self.device)

                # 前向传播
                user_emb = self.user_tower(user_feat)
                pos_item_emb = self.item_tower(pos_item_feat)
                neg_item_emb = self.item_tower(neg_item_feat.view(-1, item_feature_dim))

                # 计算损失
                loss = self._bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 计算最终embeddings
        self._compute_all_embeddings(user_features, item_features)

        self.is_fitted = True

        if verbose:
            print("\n训练完成!")

        return self

    def _compute_all_embeddings(
        self,
        user_features: Dict[int, np.ndarray],
        item_features: Dict[int, np.ndarray]
    ) -> None:
        """计算所有用户和物品的embedding"""
        self.user_tower.eval()
        self.item_tower.eval()

        with torch.no_grad():
            # 用户 embeddings
            for user_id, features in user_features.items():
                x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                emb = self.user_tower(x).squeeze(0).cpu().numpy()
                self.user_embeddings[user_id] = emb

            # 物品 embeddings
            for item_id, features in item_features.items():
                x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                emb = self.item_tower(x).squeeze(0).cpu().numpy()
                self.item_embeddings[item_id] = emb

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """获取用户embedding"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        return self.user_embeddings.get(user_id)

    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """获取物品embedding"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        return self.item_embeddings.get(item_id)

    def score(self, user_id: int, item_id: int) -> float:
        """计算用户对物品的偏好分数"""
        user_emb = self.get_user_embedding(user_id)
        item_emb = self.get_item_embedding(item_id)

        if user_emb is None or item_emb is None:
            return 0.0

        return float(np.dot(user_emb, item_emb))

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """为用户推荐物品"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            return []

        # 计算与所有物品的相似度
        scores = []
        for item_id, item_emb in self.item_embeddings.items():
            if exclude_rated and item_id in self.user_rated_items.get(user_id, set()):
                continue

            score = float(np.dot(user_emb, item_emb))
            scores.append((item_id, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def save(self, filepath: str) -> None:
        """保存模型

        Args:
            filepath: 保存路径 (.pt 文件)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练")

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        save_dict = {
            "user_embeddings": torch.tensor([self.user_embeddings[i] for i in sorted(self.user_embeddings.keys())]),
            "user_ids": torch.tensor(sorted(self.user_embeddings.keys())),
            "item_embeddings": torch.tensor([self.item_embeddings[i] for i in sorted(self.item_embeddings.keys())]),
            "item_ids": torch.tensor(sorted(self.item_embeddings.keys())),
            "user_tower_state": self.user_tower.state_dict(),
            "item_tower_state": self.item_tower.state_dict(),
            "embedding_dim": self.embedding_dim,
            "user_hidden_dims": self.user_hidden_dims,
            "item_hidden_dims": self.item_hidden_dims,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "feature_encoders": self.feature_encoders,
            "user_rated_items": dict(self.user_rated_items),
        }

        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = "auto") -> "TwoTowerModel":
        """加载模型"""
        data = torch.load(filepath, map_location=torch.device("cpu"))

        # 创建实例
        instance = cls(
            embedding_dim=data["embedding_dim"],
            user_hidden_dims=data["user_hidden_dims"],
            item_hidden_dims=data["item_hidden_dims"],
            device=device
        )

        # 恢复状态
        instance.num_users = data["num_users"]
        instance.num_items = data["num_items"]
        instance.is_fitted = True
        instance.feature_encoders = data["feature_encoders"]
        instance.user_rated_items = defaultdict(set, {k: set(v) for k, v in data["user_rated_items"].items()})

        # 恢复模型
        user_feature_dim = 23  # age(1) + gender(1) + occupation(21)
        item_feature_dim = 19  # genre(19)

        instance._build_model(user_feature_dim, item_feature_dim)
        instance.user_tower.load_state_dict(data["user_tower_state"])
        instance.item_tower.load_state_dict(data["item_tower_state"])
        instance.user_tower.to(instance.device)
        instance.item_tower.to(instance.device)
        instance.user_tower.eval()
        instance.item_tower.eval()

        # 恢复 embeddings
        user_ids = data["user_ids"].tolist()
        item_ids = data["item_ids"].tolist()
        user_embs = data["user_embeddings"].numpy()
        item_embs = data["item_embeddings"].numpy()

        for i, uid in enumerate(user_ids):
            instance.user_embeddings[int(uid)] = user_embs[i]

        for i, iid in enumerate(item_ids):
            instance.item_embeddings[int(iid)] = item_embs[i]

        return instance


if __name__ == "__main__":
    from dataset import load_movielens_100k

    # 加载数据
    data = load_movielens_100k()

    print("=" * 50)
    print("Two-Tower 模型示例 (PyTorch)")
    print("=" * 50)
    print(f"用户数: {len(data['users'])}")
    print(f"电影数: {len(data['items'])}")
    print(f"评分数: {len(data['ratings'])}")

    # 训练模型
    print("\n开始训练 Two-Tower 模型...")
    model = TwoTowerModel(
        embedding_dim=64,
        user_hidden_dims=[128, 64],
        item_hidden_dims=[128, 64],
        learning_rate=0.001,
        device="auto"
    )
    model.fit(
        users_df=data["users"],
        items_df=data["items"],
        ratings_df=data["ratings"],
        epochs=10,
        batch_size=256,
        num_negatives=4,
        verbose=True
    )

    # 获取 embedding
    user_id = 1
    item_id = 1
    user_emb = model.get_user_embedding(user_id)
    item_emb = model.get_item_embedding(item_id)
    score = model.score(user_id, item_id)

    print(f"\n用户 {user_id} 的 embedding 维度: {user_emb.shape}")
    print(f"电影 {item_id} 的 embedding 维度: {item_emb.shape}")
    print(f"用户对电影的偏好分数: {score:.4f}")

    # 推荐
    recommendations = model.recommend(user_id, top_k=5)
    print(f"\n为用户 {user_id} 推荐:")
    for rec_item_id, rec_score in recommendations:
        movie_name = data["items"][data["items"]["movie_id"] == rec_item_id]["title"].values[0]
        print(f"  {movie_name}: {rec_score:.4f}")

    # 保存模型
    model.save("models/twotower_model.pt")
