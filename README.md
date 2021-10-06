# Bornholm_data_heat_forecasting

In the whole project, the day-ahead hourly heat load forecasting is made based on the available datasets of heat load time series and numerical weather foreasts (inclusing outdoor temperature, wind speed, solar radiance, precipitation and humidity). In the end, online learning methods outperform AI methods. First, feature extraction and selection are both considered. Specifically, the techniques for time-series analysis are applied such as time-frequency domain analysis, ACF and PACF analysis. In this regard, the sine and cosine of weekly and daily periods are added as time-related variables. In addition, considering the strong 3-lag auto-correlation of the time series, therefore, numerical weather forecasts related variables and time-related variables are considered their time lags, including t-2, t-1, and t at time t. As for the heat load time series, only t-24 and t-24 are considered at time t. Therefore, all the considered variables above are used for feature selection. The selected features are used for AI methods and online learning methods. Artificial intelligence (AI) methods are applied for heat load forecasting, including 1) linear regression model, SVR, decision regression tree, SGD, Extratree, Adaboost and Gradient boost ensemble models from Sk-learn package and 2) neural networks models from Tensorflow package such as NN-Linear, NN-Dense, LSTM and CNN. The conclusion from the dataset tells that persistent model is a strong benchmark and hard to beat on average. Nonlinear regression models provide the best performance, specifically, NN-Dense. This suggests that nonlinear regression or expression is required in the models. Therefore, online learning methods (considering the yesterday heat load as input) with nonlinear transformation onto the heat load are proposed under RLS and RML frameworks. Online forecasting methods for heat load are built under recursive least squares (RLS) and recursive maximum likelihood (RML) frameworks. The hyperparameters are optimized using cross-validation methods. The performance of the algorithms are evaluated using RMSE, MAE and CRPS and their improvements compared to a benchmark. In the related .py files,

In this work, we also added the steps before modeling, including data fetching and cleaning.
# heat load data
1. Bornholm_meter_fetch.py is to fetch around 4000 meters data from energydataDK via API;
2. Bornholm_meter_prepare.py is to align the time stamps for different time series, remove the outliers;
3. Bornholm_meter_select.py is to rule out the meters with too many missing data samples and to fill the missing values for remained ones;
# weather data
4. Bornholm_data_weather_fetch.py is to fetch the weather forecasts from norwegian meterological institute via API;
5. Bornholm_weather_aggregate.py is to aggregate the monthly data into a single file and to fill the missing data; (cleaning is not required since they are forecasts);
6. Bornholm_data_aggregation_zones.py is to aggregate heat and weather data into a singel file;
# data modeling
7. helpers.py provides all the functions and global variables are required for Bornholm_forecast_sklearn.py, Bornholm_forecast_sklearn_1.py and Bornholm_forecast_feature_selection.py;
8. helpers_online.py provides all the functions and global variables for Bornholm_forecast_online.py and Bornholm_forecast_online_hyperpara.py;
9. Bornholm_forecast_feature_selection.py is to present the feature selection results;
10. Bornholm_forecast_sklearn is for neural network based models by using TensorFlow package;
11. Bornholm_forecast_sklearn1 is for machine learning models by using SK-learn package;
12. Bornholm_forecast_online_hyperpara is to get the optimal hyperparameters by evaluating the performance on the validation dataset;
13. Bornholm_forecast_online is for RLS and RML online learning models.

