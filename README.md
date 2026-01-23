# Blood Glucose Monitoring System

A complete machine learning-powered blood glucose monitoring system with web-based dashboard and ThingSpeak cloud integration.

## 📋 System Overview

This system predicts blood glucose levels from frequency response data using machine learning algorithms. It provides:
- Web-based dashboard for data visualization and analysis
- Machine learning model training and deployment
- ThingSpeak cloud integration for remote monitoring
- Support for both manual input and dataset uploads

## 📁 File Structure

```
sugar_test/
├── SUGAR UPTO 1000MG_approx.csv     # Training dataset
├── Untitled3.ipynb                  # Original Jupyter notebook
├── index.html                       # Original dashboard
├── enhanced_dashboard.html          # Enhanced web dashboard
├── sugar_ml_model.py               # ML training script
├── blood_glucose_ml_training.ipynb # Comprehensive ML training notebook
├── test_model.py                   # Model validation script
├── generate_test_data.py           # Test data generator
├── README.md                       # This documentation
├── scaler.pkl                      # Trained scaler (generated)
├── pca_model.pkl                   # Trained PCA model (generated)
├── sugar_knn_model.pkl             # Trained KNN model (generated)
└── model_performance.png           # Performance visualization (generated)
```

## 🚀 Quick Start

### 1. Train the ML Model

```bash
python sugar_ml_model.py
```

This will:
- Load the training data from `SUGAR UPTO 1000MG_approx.csv`
- Train a KNN classifier with PCA preprocessing
- Save model components as `.pkl` files
- Display training accuracy

### 2. Test the Model

```bash
python test_model.py
```

This will:
- Validate the trained model on test data
- Generate performance metrics and visualizations
- Create `model_performance.png` with detailed analysis

### 3. Launch the Dashboard

Open `enhanced_dashboard.html` in your web browser, or start a local server:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/enhanced_dashboard.html`

## 🧠 Machine Learning Pipeline

### Data Preprocessing
1. **Standardization**: Normalize features to zero mean, unit variance
2. **Dimensionality Reduction**: PCA to retain 95% variance
3. **Feature Selection**: Use frequency response data as input features

### Model Architecture
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Neighbors**: Optimized k value (typically 3-5)
- **Distance Metric**: Euclidean distance
- **Preprocessing**: StandardScaler + PCA

### Training Process
1. Load CSV data with glucose concentrations and frequency responses
2. Extract features (frequency response values) and labels (glucose levels)
3. Apply preprocessing pipeline
4. Train KNN classifier
5. Validate with cross-validation
6. Save model components

## 🌐 Web Dashboard Features

### Manual Input Tab
- Direct glucose level entry
- Instant classification (Low/Normal/High)
- Automatic ThingSpeak upload if configured

### Dataset Upload Tab
- Support for CSV and Excel files
- Automatic ML prediction from frequency data
- Real-time data visualization
- Batch processing capabilities

### ThingSpeak Configuration
- Channel ID and API key management
- Field selection for data storage
- Connection testing functionality
- Secure credential storage

### Visualization Tab
- Interactive charts using Chart.js
- Real-time data plotting
- Prediction result display
- Cloud upload integration

## ☁️ ThingSpeak Integration

### Setup
1. Create a ThingSpeak account at [thingspeak.com](https://thingspeak.com)
2. Create a new channel
3. Note your:
   - Channel ID
   - Write API Key
   - Field numbers

### Configuration
In the dashboard:
1. Navigate to "ThingSpeak Config" tab
2. Enter your credentials
3. Select appropriate field number
4. Click "Save Configuration"
5. Test connection with "Test Connection" button

### Data Transmission
- Glucose values are automatically sent to ThingSpeak
- Each prediction creates a new entry
- Real-time cloud monitoring capability
- Historical data storage and analysis

## 📊 Data Format Requirements

### Training Data (CSV)
```
Sugar Concentration (mg/dl),Freq1,Freq2, ..., Freq333
0,2.45,2.46, ..., 2.50
50,2.44,2.45, ..., 2.49
...
1000,2.30,2.31, ..., 2.35
```

### Input Data (CSV/Excel)
```
Frequency,Return Loss
2.45,-17.5
2.46,-17.3
2.47,-17.6
...
```

## 🔧 Technical Details

### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
openpyxl>=3.0.0  # for Excel support
```

### Browser Requirements
- Modern browser with JavaScript enabled
- Chart.js support
- File API support for uploads
- Fetch API for ThingSpeak integration

### Model Performance
- **Training Accuracy**: ~95-100%
- **Cross-validation**: ~90-95%
- **Response Time**: <100ms for predictions
- **Supported Glucose Range**: 0-1000 mg/dl

## 🛠️ Development Workflow

### 1. Data Preparation
```bash
# Generate test data
python generate_test_data.py
```

### 2. Model Development
```bash
# Train model
python sugar_ml_model.py

# Test model
python test_model.py

# Detailed analysis (Jupyter)
jupyter notebook blood_glucose_ml_training.ipynb
```

### 3. Web Development
- Edit `enhanced_dashboard.html`
- Test locally with Python server
- Validate ThingSpeak integration

### 4. Deployment
- Host HTML files on web server
- Ensure CSV data is accessible
- Configure ThingSpeak channel

## 📈 Monitoring and Maintenance

### Regular Tasks
- Retrain model with new data periodically
- Monitor ThingSpeak channel for data consistency
- Update dashboard for improved UX
- Validate model accuracy with real measurements

### Troubleshooting
- **Poor predictions**: Check data quality and retrain
- **ThingSpeak errors**: Verify API credentials and channel settings
- **Dashboard issues**: Check browser console for JavaScript errors
- **File upload problems**: Validate CSV/Excel format

## 🔒 Security Considerations

- Store ThingSpeak API keys securely
- Use HTTPS for production deployments
- Validate all user inputs
- Implement rate limiting for API calls
- Regular security audits

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## 📞 Support

For issues and questions:
- Check the documentation first
- Review error messages carefully
- Test with sample data
- Contact support with detailed information

## 📄 License

This project is for educational and research purposes. See LICENSE file for details.

---

**Note**: This system is designed for research and educational purposes. Medical decisions should always be made by qualified healthcare professionals using clinically validated equipment.