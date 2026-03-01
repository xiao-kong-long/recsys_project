"""MovieLens 数据集读取工具"""

import pandas as pd
from typing import Dict, List, Optional


def load_occupations(filepath: str = "data/ml-100k/u.occupation") -> List[str]:
    """加载职业枚举列表

    Returns:
        21个职业名称的列表
    """
    with open(filepath, "r", encoding="latin-1") as f:
        return [line.strip() for line in f if line.strip()]


def load_users(filepath: str = "data/ml-100k/u.user") -> pd.DataFrame:
    """加载用户数据

    格式: user id | age | gender | occupation | zip code

    Returns:
        DataFrame, occupation 保持原始字符串类型
    """
    column_names = ["user_id", "age", "gender", "occupation", "zip_code"]
    return pd.read_csv(
        filepath,
        sep="|",
        names=column_names,
        encoding="latin-1"
    )


def encode_occupation_onehot(
    users_df: pd.DataFrame,
    occupations: Optional[List[str]] = None
) -> pd.DataFrame:
    """将用户的 occupation 属性编码为 one-hot 向量

    Args:
        users_df: 用户 DataFrame
        occupations: 可选的职业列表，如不提供则从文件读取

    Returns:
        包含 occupation one-hot 列的 DataFrame (user_id + 21个职业列)
    """
    if occupations is None:
        occupations = load_occupations()

    # 创建 one-hot 编码
    onehot_data = {}
    for user_idx, row in users_df.iterrows():
        user_id = row["user_id"]
        occ = row["occupation"]

        onehot_row = {f"occ_{occ}": 1 for occ in occupations}
        onehot_row["user_id"] = user_id
        onehot_data[user_idx] = onehot_row

    result = pd.DataFrame.from_dict(onehot_data, orient="index")
    # 将缺失值填充为 0
    result = result.fillna(0).astype(int)

    return result


def load_items(filepath: str = "data/ml-100k/u.item") -> pd.DataFrame:
    """加载电影数据

    格式: movie id | movie title | release date | video release date | IMDb URL | genre1...genre19

    Returns:
        DataFrame, genre 列已经是 multi-hot 格式 (0/1)
    """
    genre_names = get_genre_names()

    column_names = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url"
    ] + genre_names

    return pd.read_csv(
        filepath,
        sep="|",
        names=column_names,
        encoding="latin-1"
    )


def load_ratings(filepath: str = "data/ml-100k/u.data") -> pd.DataFrame:
    """加载评分数据

    格式: user id | item id | rating | timestamp
    """
    column_names = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(
        filepath,
        sep="\t",
        names=column_names,
        encoding="latin-1"
    )


def load_movielens_100k(base_path: str = "data/ml-100k") -> Dict[str, pd.DataFrame]:
    """加载 MovieLens 100K 数据集的所有数据

    Returns:
        包含 users, items, ratings 的字典
    """
    return {
        "users": load_users(f"{base_path}/u.user"),
        "items": load_items(f"{base_path}/u.item"),
        "ratings": load_ratings(f"{base_path}/u.data")
    }


def load_movielens_train_test(base_path: str = "data/ml-100k", split: str = "ua") -> Dict[str, pd.DataFrame]:
    """加载 MovieLens 100K 数据集，返回训练集和测试集

    Args:
        base_path: 数据集路径
        split: 划分方式，"ua" 或 "ub"

    Returns:
        包含 users, items, train_ratings, test_ratings 的字典
    """
    return {
        "users": load_users(f"{base_path}/u.user"),
        "items": load_items(f"{base_path}/u.item"),
        "train_ratings": load_ratings(f"{base_path}/{split}.base"),
        "test_ratings": load_ratings(f"{base_path}/{split}.test")
    }


def get_genre_names() -> List[str]:
    """获取电影类型名称列表

    Returns:
        19个电影类型名称
    """
    return [
        "unknown", "action", "adventure", "animation", "children",
        "comedy", "crime", "documentary", "drama", "fantasy",
        "film-noir", "horror", "musical", "mystery", "romance",
        "sci-fi", "thriller", "war", "western"
    ]


def get_movie_genres(item_row: pd.Series) -> List[str]:
    """从电影数据行中获取该电影所属的类型列表

    Args:
        item_row: items DataFrame 中的一行

    Returns:
        电影类型名称列表
    """
    genres = get_genre_names()
    return [genre for genre in genres if item_row.get(genre, 0) == 1]


def get_user_occupation_names() -> List[str]:
    """获取所有职业名称列表

    Returns:
        21个职业名称
    """
    return load_occupations()


if __name__ == "__main__":
    # 示例用法
    data = load_movielens_100k()

    print("=" * 50)
    print("MovieLens 100K 数据集概览")
    print("=" * 50)
    print(f"用户数: {len(data['users'])}")
    print(f"电影数: {len(data['items'])}")
    print(f"评分数: {len(data['ratings'])}")

    print("\n" + "=" * 50)
    print("用户数据示例 (原始 occupation)")
    print("=" * 50)
    print(data["users"][["user_id", "age", "gender", "occupation"]].head(5))

    print("\n" + "=" * 50)
    print("用户职业 One-Hot 编码示例")
    print("=" * 50)
    users_onehot = encode_occupation_onehot(data["users"])
    print(users_onehot.head(5))
    print(f"\nOne-Hot 列数: {len(users_onehot.columns) - 1}")

    print("\n" + "=" * 50)
    print("电影数据示例 (Genre Multi-Hot)")
    print("=" * 50)
    item_sample = data["items"][["movie_id", "title"] + get_genre_names()[:5]].head(3)
    print(item_sample)

    print("\n" + "=" * 50)
    print("电影类型分布")
    print("=" * 50)
    genre_counts = data["items"][get_genre_names()].sum().sort_values(ascending=False)
    print(genre_counts.head(10))

    print("\n" + "=" * 50)
    print("职业分布")
    print("=" * 50)
    occ_counts = data["users"]["occupation"].value_counts()
    print(occ_counts)