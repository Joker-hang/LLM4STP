import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


def extract_critical_rows(group):
    """
    提取每个时间组的首尾行、最大值行和最小值行

    Args:
        group (DataFrame): 按日期分组后的 DataFrame

    Returns:
        DataFrame: 合并后的关键行数据
    """
    group = group.sort_values('date')
    first_row = group.iloc[0]
    last_row = group.iloc[-1]
    max_row = group.loc[group['OT'].idxmax()]
    min_row = group.loc[group['OT'].idxmin()]

    combined_rows = pd.concat(
        [
            pd.DataFrame([first_row]),
            pd.DataFrame([last_row]),
            pd.DataFrame([max_row]),
            pd.DataFrame([min_row])
        ],
        ignore_index=True
    )
    return combined_rows


# 读取文件
file_path = 'ETTh1.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# max_ot, min_ot = df['OT'].max(), df['OT'].min()

# 转换 'date' 列为日期时间类型
df['date'] = pd.to_datetime(df['date'])

# 按天分组
grouped = df.groupby(df['date'].dt.date)

# 应用聚合方法
result = grouped.apply(extract_critical_rows)

# 重置索引并排序
result.reset_index(drop=True, inplace=True)

# 确保 'date' 列格式正确
result['date'] = pd.to_datetime(result['date'])

# 按时间排序
result = result.sort_values('date')

# 输出结果
# print(result)

# 保存到新的 CSV 文件
result.to_csv('processed_data.csv', index=False)


def generate_matrix(data):
    # 初始化10x20矩阵
    matrix = np.zeros((10, 20), dtype=int)

    # 计算区间边界
    min_ot, max_ot = min(data), max(data)
    interval = (max_ot - min_ot) / 10

    # Bresenham直线算法
    def bresenham(x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((y0, x0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    # 坐标映射
    coordinates = []
    for col, value in enumerate(data):
        row = min(int((value - min_ot) / interval), 9)  # 映射到0-9行
        coordinates.append((row, col))
        matrix[row, col] = 1

    # 生成连接线
    for i in range(len(coordinates) - 1):
        y0, x0 = coordinates[i]
        y1, x1 = coordinates[i + 1]
        points = bresenham(x0, y0, x1, y1)
        for (y, x) in points:
            if 0 <= y < 10 and 0 <= x < 20:
                matrix[y, x] = 1

    return matrix


grouped_5 = result.groupby(pd.Grouper(key='date', freq='5D'))
for group_name, group_data in grouped_5:
    print(f"分组 {group_name}:")
    print(group_data)
    data = np.array(group_data['OT'])
    matrix = generate_matrix(data)
    matrix_ = matrix.reshape(matrix.shape[0], -1, 4).any(axis=2).astype(int)
    print(matrix)

