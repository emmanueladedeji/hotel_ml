## Intro
----------------------
The script `model.py` gathers data on booking.com journeys and trains a multi-classification model with the target of being able to predict the user's next journey.

The data and challenge was created by booking.com. More information can be found on their github repository: ![Multi-Destination Trip recommendationat booking.com](https://github.com/bookingcom/ml-dataset-mdt)

The intended



## Dataset
-------------------
The training dataset consists of over a million (1,166,835) of anonymized hotel reservations, based on real data, with the following features:
- user_id - User ID
- checkin - Reservation check-in date
- checkout - Reservation check-out date- created_date - Date when the reservation was made
- affiliate_id - An anonymized ID of affiliate channels where the booker came from (e.g. direct, some third party referrals, paid search engine, etc.)
- device_class - desktop/mobile
- booker_country - Country from which the reservation was made (anonymized)
- hotel_country - Country of the hotel (anonymized)
- city_id - city_id of the hotel's city (anonymized)
- utrip_id - Unique identification of user's trip (a group of multi-destinations bookings within the same trip).

Each reservation is a part of a customer's trip (identified by utrip_id) which includes consecutive reservations.
The evaluation dataset is constructed similarly (378,667 reservations), however the city_id (and the country) of the final reservation of each trip is concealed and requires a prediction.

## Key Areas:

### Feature Engineering
The purpose of deriving new features based on the core dataset is to try to provide the model with more predictive power.

The data was aggregated to a user level and several key features were created through the aggregation using the following code (can be seen in ![model.py](https://github.com/emmanueladedeji/hotel_ml/blob/main/model.py)):

```py
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
```

These features were then renamed for readability and traceability purposes:

```py
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
```

### Model Training

The model was trained using the XGBoost framework. This part of the `model.py` code trains the model:

```py
# train model
param_dist = {'objective':'multi:softmax', 'n_estimators': 20}
clf = xgb.XGBClassifier(**param_dist)

clf.fit(X, y,
        eval_metric='logloss',
        verbose=True)
```

### Test and Validation

The model's predictions were validated against the test set using basic model metrics such as precision, recall and F1-score using the following code (makes use of scikit-learn):

```py
# predict and evaluate on test
preds = clf.predict(test_agg_df.drop('hotel_country', axis=1))
print(classification_report(test_agg_df['hotel_country'], preds))
```

Here is a snippet of the performance on the test set:

```py
precision    recall  f1-score   support

                   Absurdistan       0.00      0.00      0.00         5
                      Aldorria       0.00      0.00      0.00        32
                       Aldovia       0.00      0.00      0.00       797
Altis and Stratis, Republic of       0.00      0.00      0.00        61
                       Alvonia       0.00      0.00      0.00      2145
                   Angrezi Raj       0.00      0.00      0.00         2
                      Aslerfan       0.00      0.00      0.00       197
                      Atlantis       0.00      0.00      0.00       251
                       Axphain       0.07      0.00      0.01      1643
                      Bacteria       0.00      0.00      0.00        36
                        Bahari       0.00      0.00      0.00        10
                       Bahavia       0.00      0.00      0.00        47
                       Baltish       0.00      0.00      0.00         8
                      Bandaria       0.00      0.00      0.00       153
                      Bangalla       0.00      0.00      0.00        69
                      Bartovia       0.00      0.00      0.00       590
                        Basran       0.00      0.00      0.00         1
                        Bialya       0.00      0.00      0.00         2
                      Bolumbia       0.00      0.00      0.00       107
                      Borginia       0.00      0.00      0.00      2906
                    Borostyria       0.00      0.00      0.00        34
                       Bozatta       0.00      0.00      0.00      2302
         Braganza, Dominion of       0.00      0.00      0.00        36
                   Brobdingnag       0.00      0.00      0.00        29
                   Bruzundanga       0.00      0.00      0.00         7
                        Bultan       0.00      0.00      0.00       229
                       Buranda       0.00      0.00      0.00        98
                  Carjackistan       0.00      0.00      0.00       502
                     Carpathia       0.00      0.00      0.00      1279
                     Chernarus       0.00      0.00      0.00        28
              Coalition States       0.00      0.00      0.00        74
                  Cobra Island       0.48      0.98      0.64      7269
   Congaree Socialist Republic       0.00      0.00      0.00        45
                    Costa Luna       0.00      0.00      0.00        10
                          Danu       0.00      0.00      0.00         2
                        Datlof       0.00      0.00      0.00        18
                    Dawsbergen       0.00      0.00      0.00      1140
                  Drusselstein       0.00      0.00      0.00      1127
                        Durhan       0.00      0.00      0.00         2
                        Edonia       0.00      0.00      0.00       305
                      El Othar       0.00      0.00      0.00        12
                       Elbonia       0.00      0.00      0.00      2766
                       Eurasia       0.00      0.00      0.00        50
                    Feldenberg       0.00      0.00      0.00        10
                  Flausenthurm       0.00      0.00      0.00        20
                        Florin       0.00      0.00      0.00         1
                   Fook Island       0.46      0.94      0.61      7262
                      Franchia       0.00      0.00      0.00        15
                     Freedonia       0.00      0.00      0.00         3
                       Genosha       0.00      0.00      0.00        94
                  Glubbdubdrib       0.03      0.04      0.04      4781
                        Gondal       0.35      0.82      0.49      5974
                 Grand Fenwick       0.00      0.00      0.00        28
                     Graustark       0.00      0.00      0.00         1
                     Graznavia       0.00      0.00      0.00        20
                    Grenyarnia       0.00      0.00      0.00       255
                    Grinlandia       0.00      0.00      0.00        16
                         Halla       0.00      0.00      0.00        52
        Holy Britannian Empire       0.03      0.00      0.00      1974
                         Idris       0.00      0.00      0.00       368
                       Illyria       0.00      0.00      0.00         3
                         Illéa       0.00      0.00      0.00         3
                   Isla Island       0.00      0.00      0.00        48
                  Isle of Fogg       0.00      0.00      0.00        23
                       Kahndaq       0.00      0.00      0.00         1
                      Kamistan       0.00      0.00      0.00       452
                        Kangan       0.02      0.02      0.02      3086
                        Kasnia       0.00      0.00      0.00      1184
                      Kazahrus       0.00      0.00      0.00        84
                      Khurain        0.00      0.00      0.00        31
                   Kumbolaland       0.00      0.00      0.00       334
                         Kyrat       0.00      0.00      0.00        12
                      Latveria       0.00      0.00      0.00         1
                      Laurania       0.00      0.00      0.00        48
                      Leutonia       0.00      0.00      0.00      1197
                      Lilliput       0.00      0.00      0.00       196
                      Lovitzna       0.00      0.00      0.00         2
                        Lugash       0.00      0.00      0.00        17
                Marina Venetta       0.00      0.00      0.00       157
                     Marshovia       0.00      0.00      0.00        13
           Merania, Kingdom of       0.00      0.00      0.00       251
                      Molvanîa       0.00      0.00      0.00         2
                      Mundania       0.00      0.00      0.00        15
                         Mypos       0.00      0.00      0.00       252
                       Nairomi       0.00      0.00      0.00         3
                       Nambutu       0.00      0.00      0.00         1
                        Naruba       0.00      0.00      0.00        80
                Nerdocrumbesia       0.00      0.00      0.00         2
                      Nevoruss       0.00      0.00      0.00       968
                       Norland       0.00      0.00      0.00       113
          North American Union       0.00      0.00      0.00         7
                   Nova Africa       0.00      0.00      0.00      1082
                    Novistrana       0.00      0.00      0.00       490
     Novoselic, The Kingdom of       0.00      0.00      0.00        33
                       Oceania       0.00      0.00      0.00       470
                     Osterlich       0.00      0.00      0.00       975
                  Outer Heaven       0.00      0.00      0.00         2
                      Palombia       0.00      0.00      0.00        98
                         Panem       0.00      0.00      0.00         5
                       Patusan       0.00      0.00      0.00      1207
Penguina (Lîle des Pingouins)        0.00      0.00      0.00         2
                     Phaic Tăn       0.00      0.00      0.00        13
                     Poictesme       0.00      0.00      0.00        10
                    Pokolistan       0.00      0.00      0.00         1
                      Pokrovia       0.00      0.00      0.00       106
                    Polrugaria       0.00      0.00      0.00         6
                   Pullamawang       0.00      0.00      0.00       407
                         Qasha       0.00      0.00      0.00        27
                        Ragaan       0.00      0.00      0.00         3
     Republic of New Rearendia       0.00      0.00      0.00         4
         Robo-Hungarian Empire       0.00      0.00      0.00       113
                      Rolisica       0.00      0.00      0.00      1364
                       Romanza       0.00      0.00      0.00         1
                  Rook Islands       0.00      0.00      0.00       189
      Russian Democratic Union       0.00      0.00      0.00        89
                   Saint Marie       0.00      0.00      0.00         3
                       Samavia       0.00      0.00      0.00       152
                   San Lorenzo       0.00      0.00      0.00        29
                  San Sombrèro       0.00      0.00      0.00       308
                 San Theodoros       0.00      0.00      0.00        10
                  Santa Prisca       0.06      0.00      0.00      1382
                       Sarkhan       0.00      0.00      0.00       126
                    Shangri-La       0.00      0.00      0.00         1
                         Slaka       0.00      0.00      0.00       277
                       Sokovia       0.00      0.00      0.00       737
           St. Georges Island        0.00      0.00      0.00         2
                         Sunda       0.00      0.00      0.00         1
                    Svenborgia       0.00      0.00      0.00        17
                      Syldavia       0.00      0.00      0.00       263
                      Sylvania       0.00      0.00      0.00       455
                      São Rico       0.00      0.00      0.00       149
                       Taronia       0.00      0.00      0.00         2
                   Tcherkistan       0.00      0.00      0.00       225
          The Devilfire Empire       0.00      0.00      0.00       550
                        Tijata       0.00      0.00      0.00         1
               Trans-Carpathia       0.00      0.00      0.00        90
                     Tsergovia       0.00      0.00      0.00         1
                     Turgistan       0.00      0.00      0.00        46
                         Uqbar       0.00      0.00      0.00        13
                        Urkesh       0.00      0.00      0.00       214
                       Urmania       0.00      0.00      0.00         1
                        Vadeem       0.00      0.00      0.00        35
                    Veyshnoria       0.00      0.00      0.00         9
                      Vulgaria       0.00      0.00      0.00         2
                        Wadiya       0.00      0.00      0.00         6
                      Wredpryd       0.00      0.00      0.00         1
                 Yellow Empire       0.00      0.00      0.00         1
                         Yerba       0.00      0.00      0.00       575
                       Yudonia       0.00      0.00      0.00       178
                      Zekistan       0.00      0.00      0.00         2
                      Zephyria       0.00      0.00      0.00        49

                      accuracy                           0.28     68496
                     macro avg       0.01      0.02      0.01     68496
                  weighted avg       0.14      0.28      0.18     68496
```

## Citing
----------------------
Booking.com Multi-Destination Trips Dataset is published as a [resource paper at SIGIR '21](https://doi.org/10.1145/3404835.3463240). Please refer to dataset in research publications in the following format:

*Dmitri Goldenberg and Pavel Levin. 2021. Booking.com Multi-Destination Trips Dataset. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21), July 11–15, 2021, Virtual Event, Canada.*
