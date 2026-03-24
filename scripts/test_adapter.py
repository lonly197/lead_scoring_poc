"""
测试数据适配器

验证新旧数据格式是否正确适配。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data.adapter import (
    detect_data_format,
    load_and_adapt_data,
    calculate_target_labels,
    derive_time_features,
    NEW_DATA_FORMAT,
    OLD_DATA_FORMAT,
)
from src.data.loader import DataLoader


def test_format_detection():
    """测试格式检测"""
    print("=" * 60)
    print("测试格式检测")
    print("=" * 60)

    # 测试新数据
    new_data_path = project_root.parent.parent / "测试数据/202603.csv"
    if new_data_path.exists():
        format_config = detect_data_format(str(new_data_path))
        is_new = format_config == NEW_DATA_FORMAT
        print(f"新数据 (202603.csv): {'✅ 正确识别为新格式' if is_new else '❌ 识别错误'}")
    else:
        print(f"新数据文件不存在: {new_data_path}")

    # 测试原数据
    old_data_path = project_root.parent.parent / "测试数据/20260308-v2.csv"
    if old_data_path.exists():
        format_config = detect_data_format(str(old_data_path))
        is_old = format_config == OLD_DATA_FORMAT
        print(f"原数据 (20260308-v2.csv): {'✅ 正确识别为原格式' if is_old else '❌ 识别错误'}")
    else:
        print(f"原数据文件不存在: {old_data_path}")


def test_new_data_loading():
    """测试新数据加载"""
    print("\n" + "=" * 60)
    print("测试新数据加载")
    print("=" * 60)

    new_data_path = project_root.parent.parent / "测试数据/202603.csv"
    if not new_data_path.exists():
        print(f"跳过: 文件不存在 {new_data_path}")
        return

    # 加载数据
    df = load_and_adapt_data(str(new_data_path))

    print(f"数据行数: {len(df):,}")
    print(f"数据列数: {len(df.columns)}")

    # 检查目标变量
    target_cols = ["到店标签_7天", "到店标签_14天", "到店标签_30天",
                   "试驾标签_14天", "试驾标签_30天", "线索评级_试驾前", "成交标签"]

    print("\n目标变量检查:")
    for col in target_cols:
        if col in df.columns:
            if col == "线索评级_试驾前":
                dist = df[col].value_counts()
                print(f"  ✅ {col}: {dict(dist)}")
            else:
                positive_rate = df[col].mean() * 100
                print(f"  ✅ {col}: 正样本率 {positive_rate:.2f}%")
        else:
            print(f"  ❌ {col}: 缺失")

    # 检查时间特征
    time_features = ["线索创建星期几", "线索创建小时"]
    print("\n时间特征检查:")
    for col in time_features:
        if col in df.columns:
            print(f"  ✅ {col}: 范围 {df[col].min()}-{df[col].max()}")
        else:
            print(f"  ❌ {col}: 缺失")


def test_data_loader():
    """测试 DataLoader 类"""
    print("\n" + "=" * 60)
    print("测试 DataLoader 类")
    print("=" * 60)

    new_data_path = project_root.parent.parent / "测试数据/202603.csv"
    if not new_data_path.exists():
        print(f"跳过: 文件不存在 {new_data_path}")
        return

    loader = DataLoader(str(new_data_path), auto_adapt=True)
    df = loader.load()

    data_format = loader.get_data_format()
    print(f"检测到的数据格式: {data_format}")

    # 获取统计信息
    stats = loader.get_basic_stats(df)
    print(f"\n数据统计:")
    print(f"  总行数: {stats['total_rows']:,}")
    print(f"  总列数: {stats['total_columns']}")
    print(f"  内存使用: {stats['memory_usage_mb']:.1f} MB")

    if "target_distribution" in stats:
        print(f"\n目标变量分布:")
        for col, dist in stats["target_distribution"].items():
            if col in ["到店标签_7天", "到店标签_14天", "到店标签_30天",
                       "试驾标签_14天", "试驾标签_30天", "成交标签"]:
                print(f"  {col}: {dist}")


def test_oot_feasibility():
    """测试 OOT 验证可行性"""
    print("\n" + "=" * 60)
    print("测试 OOT 验证可行性")
    print("=" * 60)

    new_data_path = project_root.parent.parent / "测试数据/202603.csv"
    if not new_data_path.exists():
        print(f"跳过: 文件不存在 {new_data_path}")
        return

    df = load_and_adapt_data(str(new_data_path))

    # 检查时间范围
    create_time = pd.to_datetime(df["线索创建时间"], errors='coerce')
    print(f"数据时间范围: {create_time.min()} ~ {create_time.max()}")

    # 计算14天观察窗口
    latest = create_time.max()
    cutoff = latest - pd.Timedelta(days=14)

    train_count = (create_time < cutoff).sum()
    test_count = (create_time >= cutoff).sum()

    print(f"\n14天观察窗口分析:")
    print(f"  截止日期: {cutoff.date()}")
    print(f"  训练集数量（截止日前）: {train_count:,}")
    print(f"  测试集数量（截止日后）: {test_count:,}")

    # 检查测试集的到店率
    test_df = df[create_time >= cutoff]
    if len(test_df) > 0 and "到店标签_14天" in test_df.columns:
        # 注意：测试集可能观察期不足
        print(f"\n⚠️ 注意: 测试集线索观察期不足14天")
        print(f"  测试集到店率（7天）: {test_df['到店标签_7天'].mean()*100:.2f}%")
        print(f"  建议: 等待数据沉淀后再进行 OOT 验证")


if __name__ == "__main__":
    test_format_detection()
    test_new_data_loading()
    test_data_loader()
    test_oot_feasibility()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
