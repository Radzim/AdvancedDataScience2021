import datetime
import random
from statistics import mean

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

from . import access
from . import assess


def predict_price(latitude, longitude, date, property_type, connection=None, amenities=None, box_size=3, box_years=3, amenity_distance=0.5):
    if connection is None:
        return 'No SQL connection found'

    dt = datetime.datetime.fromisoformat(date)
    epoch = datetime.datetime.utcfromtimestamp(0)
    date_as_int = int((dt - epoch).total_seconds() / 24 / 60 / 60)

    house_prices_df = access.get_house_prices(connection, (latitude, longitude), date, box_size, box_years, property_type)
    if house_prices_df.shape[0] == 0:
        return 'ERROR: found no houses in that area'
    if house_prices_df.shape[0] < 100:
        print('WARNING: possibly not enough houses for a good prediction (' + str(house_prices_df.shape[0]) + ')')
    pois_df = assess.get_pois((latitude, longitude), box_size)
    pois_concise = assess.concise_pois(pois_df)

    if amenities is None:
        all_possible_pois = []
    else:
        all_possible_pois = amenities

    local_pois = assess.find_pois_within((latitude, longitude), amenity_distance, pois_concise, all_possible_pois)
    local_pois = list(pd.Series(local_pois))
    print('Number of houses:', len(house_prices_df))
    print('Number of pois:', len(pois_df))
    print('\n\n-----------------\n\n')

    house_data = assess.prepare_dataframe_for_prediction(house_prices_df, pois_df, amenity_distance, all_possible_pois)
    house_data['date_of_transfer'] = pd.to_datetime(house_data['date_of_transfer']).astype(int) / 10 ** 9 / 24 / 60 / 60
    house_data = house_data.drop('property_type', 1)
    pois_data = house_data['pois_nearby'].apply(pd.Series)
    house_data = house_data.drop('pois_nearby', 1)

    y = house_data[['price']].values
    design = np.concatenate((house_data[['date_of_transfer', 'lattitude', 'longitude']].values, pois_data.values), axis=1)
    m_linear_basis = sm.OLS(y, design)
    results_basis = m_linear_basis.fit()
    print('Prediction basis', results_basis.summary())
    print('\n\n-----------------\n\n')

    actual_values, predicted_values = [], []
    for i in range(25):
        row_number = random.randint(0, len(house_prices_df)-1)
        actual_values.append(y[row_number][0])
        valid_date, valid_latitude, valid_longitude = house_data[['date_of_transfer', 'lattitude', 'longitude']].values[row_number]
        valid_local_pois = assess.find_pois_within((valid_latitude, valid_longitude), amenity_distance, pois_concise, all_possible_pois)
        valid_local_pois = list(pd.Series(valid_local_pois))
        y_pred_linear_valid = results_basis.get_prediction([valid_date, valid_latitude, valid_longitude] + valid_local_pois).summary_frame(alpha=0.05)
        predicted_values.append(int(y_pred_linear_valid['mean'].values[0]))
    print('Validation:\n', actual_values, '\n', predicted_values)
    print('Mean error:', str(round(mean(abs(x - y)/max(x, y) for x, y in zip(actual_values, predicted_values))*100, 2))+'%')
    if mean(abs(x - y)/max(x, y) for x, y in zip(actual_values, predicted_values)) > 0.5:
        print('WARNING: High mean error ('+str(round(mean(abs(x - y)/max(x, y) for x, y in zip(actual_values, predicted_values))*100, 2))+'%)')
    print('Correlation:', str(round(np.corrcoef(actual_values, predicted_values)[0][1], 2))+'%')
    if np.corrcoef(actual_values, predicted_values)[0][1] < 0.3:
        print('WARNING: Low correlation ('+str(round(np.corrcoef(actual_values, predicted_values)[0][1], 2))+'%'+')')
    print('\n\n-----------------\n\n')

    y_pred_linear = results_basis.get_prediction([date_as_int, latitude, longitude]+local_pois).summary_frame(alpha=0.05)
    print('Prediction:\n', y_pred_linear)
    print('Prediction value:', y_pred_linear['mean'].values[0])
    print('Prediction std err:', y_pred_linear['mean_se'].values[0])
    print('\n\n-----------------\n\n')

    sns.regplot(x="actual values", y="predicted values", data=pd.DataFrame({'actual values': actual_values, 'predicted values': predicted_values}))

    plt.tight_layout()

    return int(y_pred_linear['mean'].values[0])
