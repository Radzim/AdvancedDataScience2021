import matplotlib.pyplot as plt
import mlai.plot as plot

import osmnx as ox
from . import access
from collections import Counter


def get_pois(location, range_km, tag='amenity'):
    tags = {tag: True}
    lon_0, lon_1, lat_0, lat_1 = access.get_box(location, range_km)
    places = ox.geometries_from_bbox(lat_1, lat_0, lon_1, lon_0, tags)
    return places


def show_pois(location, range_km, tag='amenity'):
    tags = {tag: True}
    lon_0, lon_1, lat_0, lat_1 = access.get_box(location, range_km)
    places = ox.geometries_from_bbox(lat_1, lat_0, lon_1, lon_0, tags)
    graph = ox.graph_from_bbox(lat_1, lat_0, lon_1, lon_0)
    nodes, edges = ox.graph_to_gdfs(graph)
    fig, ax = plt.subplots(figsize=plot.big_figsize)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlim([lon_0, lon_1])
    ax.set_ylim([lat_0, lat_1])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    places.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()


def concise_pois(places, tag='amenity'):
    geometries = places['geometry']
    longitudes = []
    latitudes = []
    for geometry in geometries:
        latitudes.append(geometry.centroid.y)
        longitudes.append(geometry.centroid.x)
    places['latitude'] = latitudes
    places['longitude'] = longitudes
    return places[[tag, 'latitude', 'longitude']]


def get_pois_counts(pois_df, tag='amenity'):
    return Counter(pois_df[[tag]])


def find_pois_within(location, range_km, pois_concise, all_possible_pois):
    lon_0, lon_1, lat_0, lat_1 = access.get_box(location, range_km)
    pois_in_range = pois_concise.loc[(pois_concise['longitude'] >= lon_0) & (pois_concise['longitude'] <= lon_1) & (pois_concise['latitude'] >= lat_0) & (pois_concise['latitude'] <= lat_1)]
    listed_pois = pois_in_range[['amenity']].values.reshape(-1, ).tolist()
    counts = {}
    for poi in all_possible_pois:
        counts[poi] = listed_pois.count(poi)
    return counts


def prepare_dataframe_for_prediction(house_prices_df, pois_df, amenity_distance, all_possible_pois):
    columns = ['price', 'date_of_transfer', 'property_type', 'lattitude', 'longitude']
    house_prices_df_important = house_prices_df[columns]
    if len(pois_df) == 0:
        house_prices_df_important['pois_nearby'] = {}
        return house_prices_df_important
    pois_concise = concise_pois(pois_df)
    house_prices_df_important['pois_nearby'] = house_prices_df_important.apply(
        lambda row: find_pois_within((row['lattitude'], row['longitude']), amenity_distance, pois_concise, all_possible_pois), axis=1)
    return house_prices_df_important
