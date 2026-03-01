"""Two-Tower 模型训练脚本 (PyTorch 版本)"""

import argparse
import os
from dataset import load_movielens_100k
from twotower import TwoTowerModel


def main():
    parser = argparse.ArgumentParser(description="训练 Two-Tower 推荐模型 (PyTorch)")
    parser.add_argument("--data-path", type=str, default="data/ml-100k", help="数据集路径")
    parser.add_argument("--model-path", type=str, default="models/twotower_model.pt", help="模型保存路径")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding维度")
    parser.add_argument("--user-hidden", type=int, nargs="+", default=[128, 64], help="用户塔隐藏层维度")
    parser.add_argument("--item-hidden", type=int, nargs="+", default=[128, 64], help="物品塔隐藏层维度")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--num-negatives", type=int, default=4, help="负样本采样数")
    parser.add_argument("--min-rating", type=int, default=4, help="作为正样本的最低评分")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--l2-reg", type=float, default=0.01, help="L2正则化系数 (weight_decay)")
    parser.add_argument("--device", type=str, default="auto", help="计算设备: cpu/cuda/auto")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    import torch
    import numpy as np
    import random

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载数据
    print(f"从 {args.data_path} 加载数据...")
    data = load_movielens_100k(args.data_path)

    print(f"用户数: {len(data['users'])}")
    print(f"电影数: {len(data['items'])}")
    print(f"评分数: {len(data['ratings'])}")

    # 初始化模型
    model = TwoTowerModel(
        embedding_dim=args.embedding_dim,
        user_hidden_dims=args.user_hidden,
        item_hidden_dims=args.item_hidden,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        device=args.device
    )

    # 训练模型
    model.fit(
        users_df=data["users"],
        items_df=data["items"],
        ratings_df=data["ratings"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        min_rating=args.min_rating,
        verbose=True
    )

    # 确保目录存在
    os.makedirs(os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else ".", exist_ok=True)

    # 保存模型
    model.save(args.model_path)
    print(f"\n模型已保存到: {args.model_path}")
    print("运行评估: python evaluate.py --model twotower")


if __name__ == "__main__":
    main()
