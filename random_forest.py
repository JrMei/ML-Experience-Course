import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Extract time features from Date (month, weekday, day of year)
def extract_time_features(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
    if data['Date'].isna().any():
        data = data.dropna(subset=['Date'])
    
    data['Month'] = data['Date'].dt.month
    data['Weekday'] = data['Date'].dt.weekday
    data['DayOfYear'] = data['Date'].dt.dayofyear
    
    return data.drop('Date', axis=1)

# Create lagged features
def create_time_lagged_features(data, target_column, n_lags=7, n_ahead=5):
    X, y = [], []
    for i in range(n_lags, len(data) - n_ahead + 1):
        X.append(data.iloc[i-n_lags:i].values.flatten())
        y.append(data[target_column].iloc[i:i+n_ahead].values)
    return np.array(X), np.array(y)

# Check and clean data
def check_data(X):
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.mean())
    return X

# Load basin data
def load_basin_data(basin_id, n_lags=7, n_ahead=5):
    file_path = f'basin_{basin_id}.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path.")
    
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError(f"File {file_path} is empty.")
        
        data = extract_time_features(data)
        data = check_data(data)
        
        X, y = create_time_lagged_features(data, target_column='Discharge', n_lags=n_lags, n_ahead=n_ahead)
        return X, y
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"No data in file {file_path} or wrong column names.")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")

# Normalization methods
def min_max_normalize(X):
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("Input contains non-numeric types, cannot normalize.")
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def z_score_normalize(X):
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("Input contains non-numeric types, cannot normalize.")
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Feature selection methods
def pearson_feature_selection(X, y, k):
    cor_list = []
    for i in range(X.shape[1]):
        cor = pearsonr(X[:, i], y)[0]
        cor_list.append(np.abs(cor))
    indices = np.argsort(cor_list)[-k:]
    return indices

def mutual_info_feature_selection(X, y, k):
    mi = mutual_info_regression(X, y)
    indices = np.argsort(mi)[-k:]
    return indices

# Visualization function
def visualize_predictions(y_true, y_pred, basin, days_ahead=1):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label=f"True discharge for day {days_ahead}")
    plt.plot(y_pred, label=f"Predicted discharge for day {days_ahead}")
    plt.xlabel("Samples")
    plt.ylabel("Discharge")
    plt.title(f"Basin {basin}: True vs Predicted Discharge for Day {days_ahead}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    basins = [1, 2, 3]  # Basin numbers
    n_lags = 7   # Use past 7 days as features
    n_ahead = 5  # Predict next 1 to 5 days
    
    normalization_methods = {
        'Min-Max Normalization': min_max_normalize,
        'Z-score Standardization': z_score_normalize
    }
    feature_selection_methods = {
        'Pearson Correlation': pearson_feature_selection,
        'Mutual Information': mutual_info_feature_selection
    }
    results = {}

    for basin in basins:
        try:
            X, y = load_basin_data(basin, n_lags=n_lags, n_ahead=n_ahead)
            best_score = float('inf')
            best_combo = None
            best_y_test = None
            best_y_pred = None

            for norm_name, norm_func in normalization_methods.items():
                X_norm = norm_func(X)

                for fs_name, fs_func in feature_selection_methods.items():
                    selected_indices = fs_func(X_norm, y[:, 0], k=5)  # Feature selection for day 1 prediction
                    X_selected = X_norm[:, selected_indices]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=0.2, random_state=42
                    )

                    # Define parameter grid for RandomForest
                    param_grid = {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }

                    rf = RandomForestRegressor(random_state=42)
                    grid_search = GridSearchCV(
                        estimator=rf,
                        param_grid=param_grid,
                        cv=5,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )

                    grid_search.fit(X_train, y_train[:, 0])  # Train only on day 1 for now
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)

                    mse = mean_squared_error(y_test[:, 0], y_pred)

                    print(f"Basin {basin}, Normalization: {norm_name}, Feature Selection: {fs_name}, MSE: {mse}")

                    if mse < best_score:
                        best_score = mse
                        best_combo = (norm_name, fs_name)
                        best_y_test = y_test[:, 0]
                        best_y_pred = y_pred

            # Save best results for this basin
            results[basin] = {
                'Best Score': best_score,
                'Best Combination': best_combo
            }
            print(f"Basin {basin} best combination: Normalization - {best_combo[0]}, Feature Selection - {best_combo[1]}, MSE: {best_score}")

            # Visualization of predictions (only for the best combination)
            if best_y_test is not None and best_y_pred is not None:
                visualize_predictions(best_y_test, best_y_pred, basin, days_ahead=1)

        except Exception as e:
            print(f"Error loading basin {basin} data: {str(e)}")

    # Output all results
    for basin, res in results.items():
        print(f"Basin {basin}: Best MSE = {res['Best Score']}, Best Combination = {res['Best Combination']}")

if __name__ == "__main__":
    main()
