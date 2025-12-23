import numpy as np
from math import radians, sin, cos, sqrt, asin

import pandas as pd


def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return 6371 * c


def vectorized_new_coords(lats, lons, dx_nm, dy_nm):
    
    R = 6371.0
    dx = dx_nm * 1.852 / R
    dy = dy_nm * 1.852 / R

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    new_lats = np.arcsin(
        np.sin(lat_rad) * np.cos(dy) +
        np.cos(lat_rad) * np.sin(dy) * np.cos(dx)
    )
    new_lons = lon_rad + np.arctan2(
        np.sin(dx) * np.sin(dy),
        np.cos(dy) - np.sin(lat_rad) * np.sin(new_lats)
    )

    return np.degrees(new_lats), np.degrees(new_lons)


def optimized_processing():
    data = pd.read_csv('dataset/TJP/TJP.csv', header=0,
                       dtype={'lat': np.float32, 'lon': np.float32})

    data = data.iloc[data.iloc[:, 0].argsort()]

    min_lat, max_lat = data.iloc[:, 3].min(), data.iloc[:, 3].max()
    min_lon, max_lon = data.iloc[:, 2].min(), data.iloc[:, 2].max()
    print((min_lat, max_lat), (min_lon, max_lon))

    lons = data.iloc[:, 2].values.astype(float)
    lats = data.iloc[:, 3].values.astype(float)

    delta_lon = np.radians(lons - min_lon)
    lat_rad = np.radians(min_lat)
    data.iloc[:, 2] = (6371 * delta_lon * np.cos(lat_rad)) / 1.852

    delta_lat = np.radians(lats - min_lat)
    data.iloc[:, 3] = (6371 * delta_lat) / 1.852

    min_lat, max_lat = data.iloc[:, 3].min(), data.iloc[:, 3].max()
    min_lon, max_lon = data.iloc[:, 2].min(), data.iloc[:, 2].max()
    print((min_lat, max_lat), (min_lon, max_lon))

    split_idx = int(len(data) * 0.6), int(len(data) * 0.8)

    # data.iloc[:split_idx[0]].to_csv('dataset/TJP/train/TJP.csv',
    #                                 index=False, header=False)
    # data.iloc[split_idx[0]:split_idx[1]].to_csv('dataset/TJP/test/TJP.csv',
    #                                             index=False, header=False)
    # data.iloc[split_idx[1]:].to_csv('dataset/TJP/val/TJP.csv',
    #                                 index=False, header=False)


if __name__ == "__main__":
    # Area    Range    Time
    # Caofeidian Port water area    Longitude: 118°25′E - 118°95′E Latitude: 38°70′N - 39°10′N    June
    # Chengshan Jiao Promontory    Longitude: 122°50′E - 123°20′E Latitude: 37°16′N - 37°75′N    June
    # Tianjin Port water area    Longitude: 117°7′E - 118°7′E Latitude: 38°7′N - 39°1′N    June

    optimized_processing()

