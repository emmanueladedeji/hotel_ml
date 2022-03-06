from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
import xgboost as xgb
import pandas as pd

# read data
train_df = pd.read_csv("train_set.csv")

# explore
train_df.shape

train_df.head()

train_df.booker_country.value_counts()
train_df.hotel_country.value_counts()
train_df.device_class.value_counts()
train_df.user_id.nunique()

train_df.dtypes

# change data type of checkin and checkout
train_df['checkin'] = pd.to_datetime(train_df['checkin'])
train_df['checkout'] = pd.to_datetime(train_df['checkout'])

# raw feature enhancing
train_df['days_stayed'] = (train_df['checkout'] - train_df['checkin']).dt.days
train_df['days_since'] = (pd.to_datetime('today') - train_df['checkin']).dt.days
train_df['subtrips'] = train_df['utrip_id'].apply(lambda x: x.split('_')[1])

train_df.sort_values('checkin', ascending=True, inplace=True)

# do aggregate feature engineering
train_df_no_target = (train_df
                        .groupby('user_id', as_index=False)
                        .apply(lambda x: x.iloc[:-1])
                    ) # removes most recent trip as this would leak data

train_agg_df = (train_df_no_target
                .groupby('user_id', as_index=False)
                .agg({
                    'booker_country': ['count', 'nunique', 'first', 'last',],
                    'hotel_country': ['nunique', 'first', 'last',],
                    'affiliate_id': ['nunique', 'first', 'last',],
                    'city_id': ['nunique', 'first', 'last',],
                    'days_stayed': ['sum', 'mean', 'first', 'last',],
                    'days_since': ['first', 'last'],
                    'device_class': ['nunique', 'first', 'last',],
                    'subtrips': 'last'
                    })
                )

# generate target variable (most recent trip)
target_train_df = (train_df
                        .groupby('user_id', as_index=False)
                        .tail(1)
                    )[['user_id', 'hotel_country']] # get most recent trip

train_agg_df.columns = [
    'user_id',
    'no_of_trips',
    'no_of_unique_booking_location',
    'first_booking_location',
    'last_booking_location',
    'no_of_unique_destinations',
    'first_destination',
    'last_destination',
    'no_of_unique_affiliates_used',
    'first_affiliate_used',
    'last_affiliate_used',
    'no_of_unique_cities',
    'first_city_destination',
    'last_city_destination',
    'total_days_on_holiday',
    'avg_days_stayed',
    'first_days_stayed',
    'last_days_stayed',
    'days_since_first_holiday',
    'days_since_last_holiday',
    'no_unique_devices_booked_from',
    'first_device_booked_from',
    'last_device_booked_from',
    'no_of_subtrips'
]

target_train_df.columns = ['user_id', 'hotel_country']

train_agg_df['no_of_subtrips'] = train_agg_df['no_of_subtrips'].astype(int)

categorical_columns = list(train_agg_df.select_dtypes(include='object'))

# merge target variable
train_agg_df = train_agg_df.merge(target_train_df, on='user_id', how='left')

# encode all categoricals
enc = OrdinalEncoder()
train_agg_df[categorical_columns] = enc.fit_transform(train_agg_df[categorical_columns])
train_agg_df.set_index('user_id', inplace=True)

# separate input and target
y = train_agg_df['hotel_country']
X = train_agg_df.drop('hotel_country', axis=1)

# train model
param_dist = {'objective':'multi:softmax', 'n_estimators': 20}
clf = xgb.XGBClassifier(**param_dist)

clf.fit(X, y,
        eval_metric='logloss',
        verbose=True)

# predict and evaluate on train
preds = clf.predict(X)
print(classification_report(y, preds))

# read in and preprocess test set
test_df = pd.read_csv("test_set.csv")

# drop NaNs, why do they exist
test_df = test_df[~test_df.hotel_country.isna()]

# change data type of checkin and checkout
test_df['checkin'] = pd.to_datetime(test_df['checkin'])
test_df['checkout'] = pd.to_datetime(test_df['checkout'])

# raw feature enhancing
test_df['days_stayed'] = (test_df['checkout'] - test_df['checkin']).dt.days
test_df['days_since'] = (pd.to_datetime('today') - test_df['checkin']).dt.days
test_df['subtrips'] = test_df['utrip_id'].apply(lambda x: x.split('_')[1])

test_df.sort_values('checkin', ascending=True, inplace=True)

# do aggregate feature engineering
test_df_no_target = (test_df
                        .groupby('user_id', as_index=False)
                        .apply(lambda x: x.iloc[:-1])
                    ) # removes most recent trip as this would leak data

test_agg_df = (test_df_no_target
                .groupby('user_id', as_index=False)
                .agg({
                    'booker_country': ['count', 'nunique', 'first', 'last',],
                    'hotel_country': ['nunique', 'first', 'last',],
                    'affiliate_id': ['nunique', 'first', 'last',],
                    'city_id': ['nunique', 'first', 'last',],
                    'days_stayed': ['sum', 'mean', 'first', 'last',],
                    'days_since': ['first', 'last'],
                    'device_class': ['nunique', 'first', 'last',],
                    'subtrips': 'last'
                    })
                )

# generate target variable (most recent trip)
target_test_df = (test_df
                        .groupby('user_id', as_index=False)
                        .tail(1)
                    )[['user_id', 'hotel_country']] # get most recent trip

test_agg_df.columns = [
    'user_id',
    'no_of_trips',
    'no_of_unique_booking_location',
    'first_booking_location',
    'last_booking_location',
    'no_of_unique_destinations',
    'first_destination',
    'last_destination',
    'no_of_unique_affiliates_used',
    'first_affiliate_used',
    'last_affiliate_used',
    'no_of_unique_cities',
    'first_city_destination',
    'last_city_destination',
    'total_days_on_holiday',
    'avg_days_stayed',
    'first_days_stayed',
    'last_days_stayed',
    'days_since_first_holiday',
    'days_since_last_holiday',
    'no_unique_devices_booked_from',
    'first_device_booked_from',
    'last_device_booked_from',
    'no_of_subtrips'
]

target_test_df.columns = ['user_id', 'hotel_country']

test_agg_df['no_of_subtrips'] = test_agg_df['no_of_subtrips'].astype(int)

# merge target variable
test_agg_df = test_agg_df.merge(target_test_df, on='user_id', how='left')

# encode all categoricals

# hard coded messy fix
for col in categorical_columns:
    test_agg_df = test_agg_df[~test_agg_df[col].isin(['Romanza', 'Takistan', 'Maltovia', 'Basran', 'Pokolistan'])]

test_agg_df[categorical_columns] = enc.transform(test_agg_df[categorical_columns])
test_agg_df.set_index('user_id', inplace=True)

# predict and evaluate on test
preds = clf.predict(test_agg_df.drop('hotel_country', axis=1))
print(classification_report(test_agg_df['hotel_country'], preds))
