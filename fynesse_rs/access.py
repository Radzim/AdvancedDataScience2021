# relevant imports
import mysql.connector
import matplotlib.pyplot as plt
import mlai.plot as plot
import osmnx as ox
import pandas as pd
import numpy as np
from math import cos, asin, sqrt, pi
import datetime


# functions to define coordinate box around point from km distance data
def distance(lat1, lon1, lat2, lon2):
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def get_box(location, range_km):
    reference_point = (52, 0)
    latitude_angle_to_km = distance(reference_point[0] - 0.5, reference_point[1], reference_point[0] + 0.5,
                                    reference_point[1])
    longitude_angle_to_km = distance(reference_point[0], reference_point[1] - 0.5, reference_point[0],
                                     reference_point[1] + 0.5)
    box_lat = range_km / latitude_angle_to_km
    box_lon = range_km / longitude_angle_to_km
    return location[1] - box_lon / 2, location[1] + box_lon / 2, location[0] - box_lat / 2, location[0] + box_lat / 2


# Function to return the list of properties in a given range, time and type, and one that returns all properties in a
# given range (regardless of time and type)
def get_house_prices(connection, location, date, range_km, range_years, property_type):
    lon_0, lon_1, lat_0, lat_1 = get_box(location, range_km)
    datetime_object = datetime.datetime.fromisoformat(date)
    timedelta_object = datetime.timedelta(days=range_years * 365 // 2)
    date_0, date_1 = str(datetime_object - timedelta_object)[:10], str(datetime_object + timedelta_object)[:10]
    return get_house_prices_inner(connection, lat_0, lat_1, lon_0, lon_1, date_0, date_1, property_type)


def get_house_prices_all(connection, location, range_km):
    lon_0, lon_1, lat_0, lat_1 = get_box(location, range_km)
    houses = get_house_prices_all_inner(connection, lat_0, lat_1, lon_0, lon_1)
    return houses


def show_house_prices(connection, location, range_km):
    lon_0, lon_1, lat_0, lat_1 = get_box(location, range_km)
    houses = get_house_prices_all_inner(connection, lat_0, lat_1, lon_0, lon_1)
    graph = ox.graph_from_bbox(lat_1, lat_0, lon_1, lon_0)
    nodes, edges = ox.graph_to_gdfs(graph)
    fig, ax = plt.subplots(figsize=plot.big_figsize)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlim([lon_0, lon_1])
    ax.set_ylim([lat_0, lat_1])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    x, y, z = houses[['longitude']].values, houses[['lattitude']].values, houses[['price']].values
    z = np.minimum(z, np.percentile(z, 90))
    plt.scatter(x, y, c=z, alpha=0.5)
    plt.tight_layout()

# These are the inner functions that send the appropriate query sent to the database with the proper values (
# described above)
def get_house_prices_inner(connection, lat_0, lat_1, lon_L, lon_H, date_L, date_H, prop_type):
    query = """
  SELECT a_t.`date_of_transfer`, pcd.`postcode`, pcd.`lattitude`, pcd.`longitude`, a_t.`property_type`, a_t.`price` FROM
  (SELECT `postcode`, `date_of_transfer`, `property_type`, `price` FROM `pp_data`
  WHERE `postcode` IN 
  (SELECT `postcode` from `postcode_data`
  WHERE `lattitude` BETWEEN {lat_0} AND {lat_1}
  AND `longitude` BETWEEN {lon_0} AND {lon_1})
  AND `date_of_transfer` BETWEEN '{date_0}' AND '{date_1}'
  AND `property_type` = '{prop_type}') a_t
  INNER JOIN
  `postcode_data` pcd
  ON (pcd.`postcode`= a_t.`postcode`)
  """
    query = query.format(lat_0=lat_0, lat_1=lat_1, lon_0=lon_L, lon_1=lon_H, date_0=date_L, date_1=date_H,
                         prop_type=prop_type)
    df = pd.read_sql(query, connection)
    for col in ['postcode', 'property_type']:
        df[col] = df[col].apply(lambda x: x.decode("utf-8"))
    return df


def get_house_prices_all_inner(connection, lat_0, lat_1, lon_0, lon_1):
    query = """
  SELECT a_t.`date_of_transfer`, pcd.`postcode`, pcd.`lattitude`, pcd.`longitude`, a_t.`property_type`, a_t.`price` FROM
  (SELECT `postcode`, `date_of_transfer`, `property_type`, `price` FROM `pp_data`
  WHERE `postcode` IN 
  (SELECT `postcode` from `postcode_data`
  WHERE `lattitude` BETWEEN {lat_0} AND {lat_1}
  AND `longitude` BETWEEN {lon_0} AND {lon_1})) a_t
  INNER JOIN
  `postcode_data` pcd
  ON (pcd.`postcode`= a_t.`postcode`)
  """
    query = query.format(lat_0=lat_0, lat_1=lat_1, lon_0=lon_0, lon_1=lon_1)
    df = pd.read_sql(query, connection)
    for col in ['postcode', 'property_type']:
        df[col] = df[col].apply(lambda x: x.decode("utf-8"))
    return df
