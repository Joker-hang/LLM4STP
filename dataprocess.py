import math
import os

import numpy as np
from PIL import Image
import pandas as pd

from geopy.distance import geodesic


def calculate_distance(coord1, coord2):
    """
    计算两个经纬度坐标之间的距离

    :param coord1: 第一个坐标，格式为 (纬度, 经度)
    :param coord2: 第二个坐标，格式为 (纬度, 经度)
    :return: 距离（单位：千米）
    """
    distance = geodesic(coord1, coord2).kilometers
    return distance


def calculate_new_coordinates(lat1, lon1, dx_nautical_miles, dy_nautical_miles):
    """
    计算给定一个点的经纬度以及另一个点在x（经度）和y（纬度）方向上的海里数后，另一个点的经纬度

    :param lat1: 原始纬度
    :param lon1: 原始经度
    :param dx_nautical_miles: 经度方向上的海里数
    :param dy_nautical_miles: 纬度方向上的海里数
    :return: 新的经纬度坐标
    """
    # 地球半径（千米）
    R = 6371.0

    # 将海里转换为千米
    dx_km = dx_nautical_miles * 1.852
    dy_km = dy_nautical_miles * 1.852

    # 将距离转换为弧度
    dx_rad = dx_km / R
    dy_rad = dy_km / R

    # 原始纬度和经度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)

    # 计算新的纬度
    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(dy_rad) + math.cos(lat1_rad) * math.sin(dy_rad) * math.cos(dx_rad))
    lat2 = math.degrees(lat2_rad)

    # 计算新的经度
    lon2_rad = lon1_rad + math.atan2(math.sin(dx_rad) * math.sin(dy_rad),
                                     math.cos(dy_rad) - math.sin(lat1_rad) * math.sin(lat2_rad))
    lon2 = math.degrees(lon2_rad)

    return lat2, lon2


# Area    Range    Time
# Caofeidian Port water area    Longitude: 118°25′E - 118°95′E Latitude: 38°70′N - 39°10′N    June
# Chengshan Jiao Promontory    Longitude: 122°50′E - 123°20′E Latitude: 37°16′N - 37°75′N    June
# Tianjin Port water area    Longitude: 117°7′E - 118°7′E Latitude: 38°7′N - 39°1′N    June

# 读取CSV文件
csv_file_path = 'dataset/CFD/CFD.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(csv_file_path, header=0)
data = data.sort_values(by=data.columns[0], ascending=True)

min_lat, max_lat = data.iloc[:, 3].min(), data.iloc[:, 3].max()  # min_lat:38.70 max_lat: 39.1
min_lon, max_lon = data.iloc[:, 2].min(), data.iloc[:, 2].max()  # min_lon:118.25 max_lon: 118.95
print((min_lat, max_lat), (min_lon, max_lon))
coordinate = (min_lat, min_lon)
for index in range(data.shape[0]):
    # 坐标
    coordinate_x = (min_lat, data.iloc[index, 2])
    distance_x = calculate_distance(coordinate_x, coordinate)
    coordinate_y = (data.iloc[index, 3], min_lon)
    distance_y = calculate_distance(coordinate_y, coordinate)
    data.iloc[index, 2] = distance_x / 1.852
    data.iloc[index, 3] = distance_y / 1.852
# 计算分割点
split_index = int(len(data) * 0.6)
split_index_2 = int(len(data) * 0.8)

min_lat_nm, max_lat_nm = data.iloc[:, 3].min(), data.iloc[:, 3].max()
min_lon_nm, max_lon_nm = data.iloc[:, 2].min(), data.iloc[:, 2].max()

print(min_lat_nm, max_lat_nm)  # 23.227774087068585
print(min_lon_nm, max_lon_nm)  # 31.30592852627403

# 拆分数据为6:2:2两部分
data_part1 = data[:split_index]
data_part2 = data[split_index:split_index_2]
data_part3 = data[split_index_2:]

# 将第一部分存储到新的CSV文件中
output_file_path1 = 'dataset/CFD/train/CFD.csv'  # 第一部分的输出文件路径
data_part1.to_csv(output_file_path1, index=False, header=None)

# 将第二部分存储到新的CSV文件中
output_file_path2 = 'dataset/CFD/test/CFD.csv'  # 第二部分的输出文件路径
data_part2.to_csv(output_file_path2, index=False, header=None)

output_file_path3 = 'dataset/CFD/val/CFD.csv'
data_part3.to_csv(output_file_path3, index=False, header=None)

print(f"数据已拆分并存储到 {output_file_path1} 和 {output_file_path2} 和 {output_file_path3}")
