import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

def train_improved_glucose_model():
    """Train improved ML model for glucose prediction using regression"""
    print("Loading sugar concentration data...")
    
    # Load the CSV data
    data = pd.read_csv("SUGAR UPTO 1000MG_approx.csv")
    
    # Extract frequency column and concentration columns
    freq_column = 'Freq [GHz]'
    conc_columns = [col for col in data.columns if col.endswith('MG') and col != freq_column]
    
    # Labels are the concentration levels
    y = np.array([int(col.replace('MG', '')) for col in conc_columns])
    
    # Extract data for each concentration
    X = []
    for col in conc_columns:
        X.append(data[col].values)
    X = np.array(X)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Concentration range: {y.min()} - {y.max()} mg/dl")
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"After PCA: {X_pca.shape[1]} components")
    
    # Try different regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale')
    }
    
    best_model = None
    best_score = -float('inf')
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        # Train model
        model.fit(X_pca, y)
        
        # Evaluate
        predictions = model.predict(X_pca)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        print(f"{name}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (R² = {best_score:.4f})")
    
    # Save the best model components
    joblib.dump(scaler, 'improved_scaler.pkl')
    joblib.dump(pca, 'improved_pca.pkl')
    joblib.dump(best_model, 'improved_glucose_model.pkl')
    
    # Test predictions
    final_predictions = best_model.predict(X_pca)
    accuracy_within_50 = np.mean(np.abs(final_predictions - y) <= 50) * 100
    accuracy_within_100 = np.mean(np.abs(final_predictions - y) <= 100) * 100
    
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: {mean_absolute_error(y, final_predictions):.2f} mg/dl")
    print(f"Root Mean Square Error: {np.sqrt(mean_squared_error(y, final_predictions)):.2f} mg/dl")
    print(f"R² Score: {r2_score(y, final_predictions):.4f}")
    print(f"Accuracy within 50 mg/dl: {accuracy_within_50:.1f}%")
    print(f"Accuracy within 100 mg/dl: {accuracy_within_100:.1f}%")
    
    return scaler, pca, best_model, X, y

def predict_glucose_concentration(frequency_data):
    """Predict sugar concentration from frequency readings using improved model"""
    try:
        # Load trained models
        scaler = joblib.load('improved_scaler.pkl')
        pca = joblib.load('improved_pca.pkl')
        model = joblib.load('improved_glucose_model.pkl')
        
        # Preprocess input (ensure it's the right shape - 401 features)
        if len(frequency_data) != 401:
            if len(frequency_data) < 401:
                # Pad with last value
                padded_data = list(frequency_data) + [frequency_data[-1]] * (401 - len(frequency_data))
                frequency_data = padded_data
            else:
                # Truncate
                frequency_data = frequency_data[:401]
        
        X_scaled = scaler.transform([frequency_data])
        X_pca = pca.transform(X_scaled)
        
        # Predict
        prediction = model.predict(X_pca)[0]
        return max(0, min(1000, prediction))  # Clamp to valid range
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None

if __name__ == "__main__":
    # Train the improved model
    scaler, pca, model, X, y = train_improved_glucose_model()
    
    # Test with sample data
    sample_data = X[5]  # 250mg sample
    prediction = predict_glucose_concentration(sample_data)
    print(f"\nSample prediction: {prediction:.1f} mg/dl")
    print(f"Actual value was: 250 mg/dl")
    print(f"Error: {abs(prediction - 250):.1f} mg/dl")