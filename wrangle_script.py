import pandas as pd
import numpy as np
import dateutil.parser as dp
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import sys
import argparse
import matplotlib.pyplot as plt
import logging
from configparser import ConfigParser
import pyproj

def calculate_statistics(old_df, new_df):
    old_df['dt'] = old_df.groupby('id')['t'].diff().values
    old_df = calculate_speed(old_df)
    new_df = calculate_speed(new_df)
    new_df_speed_by_leg = calculate_speed(new_df, 'leg_num')
    print("+++ CALCULATING STATISTICS +++")
    print("+++ ORIGINAL DATAFRAME +++")
    print(old_df.describe())
    print("+++ WRANGLED DATAFRAME +++")
    print(new_df.describe())
    show_percentage_of_rows_with_zero_diff_per_leg(old_df, 'id', "Raw dataframe groupby id")
    show_percentage_of_rows_with_zero_diff_per_leg(new_df, 'id', "Wrangled dataframe groupby id")
    show_percentage_of_rows_with_zero_diff_per_leg(new_df_speed_by_leg, 'leg_num', 'Wrangled dataframe groupby leg id')
    show_distance_travelled_over_time(old_df, "Distance travelled over time with original dataframe")
    show_distance_travelled_over_time(new_df, "Distance travelled over time with modified dataframe")
    show_distance_travelled_over_time(new_df_speed_by_leg,"Distance travelled over time with modified dataframe per leg of journey", groupby_id='leg_num')
    

def show_distance_travelled_over_time(df, title, groupby_id='id'):
    df['dt_cum'] = df.groupby(groupby_id).dt.cumsum()
    df['dist_cum'] = df.groupby(groupby_id).dist_m.cumsum()
    print(df)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    for shipid in df[groupby_id].unique():
        color = tuple(np.random.rand(1,3)[0])
        d = df[df[groupby_id]==shipid]
        ax1.plot(d.dt_cum, d.dlon, '-', color=color); ax1.set_xlabel('cumulative dt(sec)'); ax1.set_ylabel('delta distance in longitude(m)'); 
        ax2.plot(d.dt_cum, d.dlat, '-', color=color); ax2.set_xlabel('cumulative dt(sec)'); ax2.set_ylabel('delta distance in latitude(m)'); 
        ax3.plot(d.dt_cum, d.dist_cum, '-', color=color); ax3.set_xlabel('cumulative dt(sec)'); ax3.set_ylabel('cumulative distance(m)')
    fig.suptitle(title)
    plt.show()

def show_percentage_of_rows_with_zero_diff_per_leg(df, groupby_id, title):
    df = df.sort_values('t')
    df['dlat'] = df.groupby(groupby_id).lat.diff()
    df['dlon'] = df.groupby(groupby_id).lon.diff()
    lat_in_top_bin = []
    lon_in_top_bin = []
    zero_speed = []
    for g_id in df[groupby_id].unique():
        d = df[df[groupby_id]==g_id]#.dropna(subset=['speed'])
        lat_in_top_bin.append((len(d), len(d[d.dlat==0.0])/len(d)))
        lon_in_top_bin.append((len(d), len(d[d.dlon==0.0])/len(d)))
        zero_speed.append((len(d), len(d[d.speed==0.0])/len(d)))
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,20))
    lat_in_top_bin = sorted(lat_in_top_bin, key=lambda x: x[0])
    lon_in_top_bin = sorted(lon_in_top_bin, key=lambda x: x[0])
    zero_speed = sorted(zero_speed, key=lambda x: x[0])
    fig.suptitle(title)
    ax1.set_xlabel('number of records')
    ax1.set_title('Delta latitude')
    ax1.set_ylabel('percentage of zero diff rows')
    ax2.set_xlabel('number of records')
    ax2.set_title('Delta longitude')
    ax2.set_ylabel('percentage of zero diff rows')
    ax3.set_xlabel('number of records')
    ax3.set_title('speed')
    ax3.set_ylabel('percentage with zero speed')
    a = np.array(lat_in_top_bin).T
    b = np.array(lon_in_top_bin).T
    c = np.array(zero_speed).T
    ax1.scatter(a[0], a[1]*100, color="red")
    ax2.scatter(b[0], b[1]*100, color="blue")
    ax3.scatter(c[0], c[1]*100, color="green")
    plt.show()

def load_dataset():
    '''Loads the dataset and rename the columns. '''
    logger.info("+++ START +++\n+++ READING DATAFRAME FROM FILE +++")
    args = config_object['DATAFRAME']
    df = pd.read_csv(args.pop('filepath'))
    # Standardize column names
    df = df.rename(columns={args['id_column']:'id', args['lon_column']: 'lon', args['lat_column']: 'lat', args['time_column']:'t'})
    original_length = len(df)
    # Change time to number
    df = transform_time(df)
    df = convert_coords_to_utm(df)
    logger.info("+++ ORIGINAL LENGTH: " + str(original_length) + "+++\n")
    return df

def drop_dublicates(df):
    original_length = len(df)
    logger.info("+++ DROPPING ID/TIME DUBLICATES +++")
    args = dict(config_object['DATAFRAME'])
    df = df.drop_duplicates(['id', 't'])
    prev_len = len(df)
    logger.info("+++ ROWS DROPPED: " + str(original_length-prev_len) + "+++\n")
    # Drop rows that are missing coordinates
    logger.info("+++ DROPPING ROWS WHERE COORDINATES ARE MISSING +++")
    prev_len = len(df)
    df = df.dropna(subset=['lon', 'lat'], how='any')
    logger.info("+++ ROWS DROPPED:" + str(prev_len-len(df)) + "+++\n")
    return df

def drop_small_legs(df):
    ''' If timedelta between two measurements is more than the threshold defined in config file  '''
    df = df.sort_values('t')
    df['dt'] = (df['t'].groupby(df['id']).diff()).values
    interm = df[(df.dt > int(config_object['WRANGLING']['leg_gap'])) | df.dt.isna()].reset_index()['index']
    df['leg_num'] = pd.Series(interm.index, index=interm)
    df['leg_num'] = df.groupby('id').leg_num.fillna(method='ffill')
    logger.info("+++ DROPPING SHIP LEGS THAT HAVE FEWER THAN " + config_object['WRANGLING']['min_rows'] + " MEASUREMENTS +++")
    prev_len = len(df)
    df = df[df.groupby('id')['id'].transform('size')>int(config_object['WRANGLING']['min_rows'])]
    logger.info("+++ ROWS DROPPED: " + str(prev_len-len(df)) + " +++\n")
    return df

def fix_values(df):
    # Calculate the leg numbers for the original dataset. Drop small legs since they can not be fixed
    df = drop_small_legs(df)
    ship_legs = df.groupby('leg_num')
    logger.info("+++ FIXING SPEED VALUES OUTLIERS +++")
    logger.info("+++ TOTAL OF OUTLIERS: LAT-" + str(len(df[df.lat.isna()])) + " LON-" + str(len(df[df.lon.isna()])))
    # Fix values using rolling averages
    rolling_averages_lat = ship_legs.lat.rolling(10, min_periods=1, center=True).mean()
    rolling_averages_lon = ship_legs.lon.rolling(10, min_periods=1, center=True).mean()
    df.lat.fillna(rolling_averages_lat.reset_index(level=0)['lat'])
    df.lon.fillna(rolling_averages_lon.reset_index(level=0)['lon'])
    return df


def transform_time(df):
    '''Change the time column to unix epoch. If the time is already in numeric format return it as is'''
    if not is_numeric_dtype(df['t']):
        if not is_datetime64_any_dtype(df['t']):
            df['t'] = df['t'].apply(lambda x: dp.parse(x))
        # Sort values by timestamp
        df['t'] = df.t.apply(lambda x: abs(x.timestamp()))
    return df
    
def calculate_speed(df, groupby_id = 'id'):
    df = df.sort_values('t')
    # # Calculate speed of movement(in meters/sec) from one location to another
    df['dt'] = (df['t'].groupby(df[groupby_id]).diff()).values  # Diff Time in seconds
    df['dlon'] = (df['lon'].groupby(df[groupby_id]).diff()).values  # Diff Longitude in meters
    df['dlat'] = (df['lat'].groupby(df[groupby_id]).diff()).values  # Diff Latitude in meters
    df['dist_m'] = np.sqrt(df.dlon.values ** 2 + df.dlat.values ** 2)  # euclidean distance in meters
    df['speed'] = df['dist_m'] / df['dt']
    return df

def remove_speed_outliers(df):
    '''Calculate speed using timedeltas and haversine distance between two points. Uses auxiliarry calculate_speed function for fast axis wise application'''
    # iterate this process until no rows are dropped 
    iterate_speed = True
    # Drop rows where the speed excedes the allowed value
    # Remove too fast entries
    df = calculate_speed(df)
    while iterate_speed:
        old_len = df.shape[0]
        df = df.drop(df[(df.dt <= int(config_object['WRANGLING']['leg_gap'])) & (df.speed > float(config_object['WRANGLING']['max_speed']))].index)
        df = calculate_speed(df)
        new_len = df.shape[0]
        if old_len == new_len:
            iterate_speed = False
    iterate_speed = True
    # Remove too slow entries
    df = calculate_speed(df)
    while iterate_speed:
        old_len = df.shape[0]
        df = df.drop(df[(df.dt <= int(config_object['WRANGLING']['leg_gap'])) & (df.speed < float(config_object['WRANGLING']['min_speed']))].index)
        df = calculate_speed(df)
        new_len = df.shape[0]
        if old_len == new_len:
            iterate_speed = False
    return df

def convert_coords_to_utm(df):
    myProj = pyproj.Proj(init='epsg:3857')
    lon2, lat2 = myProj(df['lon'].values, df['lat'].values)
    df.update(pd.DataFrame({'lon': lon2, 'lat': lat2}))
    return df

def wrangle_dataset(df):
    # Bearing calculation function
    def calculate_bearing(lat):
        dlon = np.absolute(df.loc[lat.index, 'lon'] - df.loc[lat.index, 'lon_prev'])
        X = np.cos(df.loc[lat.index, 'lat_prev'])* np.sin(dlon)
        Y = np.cos(lat) * np.sin(df.loc[lat.index, 'lat_prev']) - np.sin(lat) * np.cos(df.loc[lat.index, 'lat_prev']) * np.cos(dlon)
        return np.degrees(np.arctan2(X,Y))
    original_length=len(df)
    original_df = df.copy()
    df = drop_dublicates(df)
    # Remove rows where the speed is above or under threshold defined in config 
    df = remove_speed_outliers(df)
    # Differentiate the legs
    df = drop_small_legs(df)
    if int(config_object['WRANGLING']['fix_values']):
        # get the indices that have changed during wrangling process
        diff_indices = original_df.index.difference(df.index)
        original_df.loc[list(diff_indices), 'lat'] = np.nan
        original_df.loc[list(diff_indices), 'lon'] = np.nan
        df = fix_values(original_df)
    '''
    if bearing:
        df['bearing'] = df.groupby('leg_num')['lat'].apply(calculate_bearing)
    '''
    logger.info("+++ TOTAL ROWS DROPPED: " + str(original_length-len(df)) + " +++")
    logger.info("---DONE---")
    return df


parser = argparse.ArgumentParser(description="Tool for data wrangling")
parser.add_argument('-v', '--verbose', help = 'show program output', default=False, action="store_true")
parser.add_argument('-s', '--statistics', help='calculate statistics on the transformation', default=False, action="store_true")

config_object = ConfigParser()
config_object.read("config.ini")
cli_args=vars(parser.parse_args())
logger = logging.getLogger(__name__)
if cli_args['verbose']:
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)
raw_dataframe = load_dataset()
fixed_dataframe = wrangle_dataset(raw_dataframe)
if cli_args['statistics']:
    calculate_statistics(raw_dataframe, fixed_dataframe)
fixed_dataframe.to_csv("out_ais.csv") #+ config_object['DATAFRAME']['filepath'])

