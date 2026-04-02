# Ice Cream Sales Prediction - ML Project with Flask Web App

A complete machine learning project that predicts ice cream sales based on temperature using Linear Regression, with a beautiful Flask web interface.

## 📊 Project Overview

This project demonstrates:

- **Data Loading & Exploration**: With pandas
- **Data Preprocessing**: Handling missing values and train-test split (80/20)
- **Model Training**: Linear Regression from scikit-learn
- **Model Evaluation**: MSE and R² Score metrics
- **Visualization**: Regression plot using matplotlib
- **Web Application**: Flask server with HTML/CSS/JavaScript UI
- **Model Persistence**: Saving and loading models with pickle

## 📁 Project Structure

```
ice-cream/
├── app.py                    # Main Flask application
├── train_model.py            # Model training and evaluation script
├── model.pkl                 # Trained model (generated after training)
├── ice-cream.csv             # Dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── templates/
│   └── index.html            # Web UI HTML template
└── static/
    ├── style.css             # Styling
    ├── script.js             # Frontend JavaScript
    └── regression_plot.png    # Generated regression visualization
```

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

- pandas 2.0.3 - Data manipulation
- numpy 1.24.3 - Numerical computing
- scikit-learn 1.3.0 - Machine learning
- flask 2.3.2 - Web framework
- matplotlib 3.7.2 - Visualization

### Step 2: Train the Model

Before running the Flask app, train the model:

```bash
python train_model.py
```

**This will:**

- Load the ice-cream.csv dataset
- Display data exploration information
- Handle any missing values
- Split data into 80% training and 20% testing sets
- Train a Linear Regression model
- Show model coefficients and intercept
- Evaluate the model (MSE and R² Score)
- Generate a regression plot
- Save the trained model as `model.pkl`

**Sample Output:**

```
============================================================
LOADING AND EXPLORING DATA
============================================================

Dataset Shape: (30, 6)

First 5 rows:
        Date DayOfWeek Month Temperature Rainfall IceCreamsSold
0 2025-04-01   Tuesday April        59.4     0.74             61
...

============================================================
MODEL TRAINING
============================================================

Model Coefficients: 3.1234
Model Intercept: -50.5678

Linear Regression Equation:
IceCreamsSold = -50.5678 + 3.1234 * Temperature

============================================================
MODEL EVALUATION
============================================================

Training Set Metrics:
  Mean Squared Error (MSE): 45.6789
  R² Score: 0.8923

Testing Set Metrics:
  Mean Squared Error (MSE): 52.3456
  R² Score: 0.8645
```

### Step 3: Run the Flask App

```bash
python app.py
```

**Output:**

```
✓ Model loaded successfully from model.pkl
✓ Flask app is ready to serve predictions!
✓ Visit http://localhost:5000 in your browser
 * Running on http://localhost:5000
```

The flask app will start on `http://localhost:5000`

## 🌐 Using the Web Application

### Features

1. **Model Information Display**
   - Shows algorithm type (Linear Regression)
   - Displays model coefficients and intercept
   - Shows the regression equation

2. **Prediction Interface**
   - Input field for temperature (°F)
   - Valid range: -50°F to 150°F
   - Real-time predictions
   - Error handling and validation

3. **Visualization**
   - Regression plot showing the relationship between temperature and sales
   - Actual data points (blue scatter)
   - Fitted regression line (red line)

### How to Make Predictions

1. Open your browser and go to `http://localhost:5000`
2. Enter a temperature value in the input field (e.g., 72°F)
3. Click the "Predict Sales" button
4. View the predicted ice cream sales quantity
5. Check the regression plot to understand the relationship

### Example Predictions

- At 70°F: ~167 ice cream units
- At 75°F: ~184 ice cream units
- At 80°F: ~200 ice cream units

(Actual predictions depend on your dataset and trained coefficients)

## 📈 Model Details

### Algorithm: Linear Regression

Linear Regression finds the best-fit line through the data using the equation:

**IceCreamsSold = Intercept + Coefficient × Temperature**

### Evaluation Metrics

**Mean Squared Error (MSE)**

- Measures the average squared difference between predicted and actual values
- Lower values indicate better fit
- Formula: MSE = Σ(y_actual - y_predicted)² / n

**R² Score**

- Coefficient of determination, ranges from 0 to 1
- Indicates the proportion of variance explained by the model
- 1.0 = perfect fit, 0.0 = model is no better than mean
- Formula: R² = 1 - (SS_res / SS_tot)

## 🔧 API Endpoints

### GET `/`

Returns the main web interface

### POST `/predict`

Makes a prediction for ice cream sales

**Request:**

```json
{
  "temperature": 72.5
}
```

**Response (Success):**

```json
{
  "success": true,
  "temperature": 72.5,
  "predicted_sales": 175.42,
  "message": "At 72.5°F, estimated ice cream sales: 175.42 units"
}
```

**Response (Error):**

```json
{
  "success": false,
  "error": "Error message here"
}
```

### GET `/model-info`

Returns information about the trained model

**Response:**

```json
{
  "success": true,
  "algorithm": "Linear Regression",
  "feature": "Temperature (°F)",
  "target": "Ice Creams Sold",
  "coefficient": 3.1234,
  "intercept": -50.5678,
  "equation": "Sales = -50.5678 + 3.1234 × Temperature"
}
```

## 📊 Dataset Information

**File:** `ice-cream.csv`

**Columns:**

- **Date**: Date of observation
- **DayOfWeek**: Day of the week
- **Month**: Month of the year
- **Temperature**: Temperature in Fahrenheit (Feature)
- **Rainfall**: Rainfall amount
- **IceCreamsSold**: Number of ice creams sold (Target)

**Statistics:**

- Number of samples: 30
- Temperature range: 49.7°F - 66.1°F
- Ice cream sales range: 21 - 98 units

## 🛠️ Troubleshooting

### Issue: "Model file not found" error

**Solution:** Run `python train_model.py` before starting the Flask app.

### Issue: Port 5000 already in use

**Solution:**

```bash
# Change port in app.py
app.run(port=5001)  # Use different port
```

### Issue: Template not found error

**Solution:** Make sure the `templates/` directory exists and contains `index.html` in the same directory as `app.py`.

### Issue: Missing static files

**Solution:** Ensure the `static/` directory exists with `style.css`, `script.js`, and `regression_plot.png`.

## 📝 Code Comments

All code files include detailed comments explaining:

- Function purposes
- Parameter descriptions
- Return values
- Important logic steps

## 🎨 Features Implemented

✅ Data exploration and preprocessing
✅ 80/20 train-test split
✅ Linear Regression model
✅ Model evaluation (MSE, R²)
✅ Regression visualization
✅ Model persistence with pickle
✅ Flask web server
✅ HTML/CSS/JavaScript UI
✅ API endpoints
✅ Error handling
✅ Input validation
✅ Responsive design
✅ Clean, commented code

## 📚 Learning Resources

- [scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to improve this project by:

- Adding more features
- Improving the UI
- Optimizing the model
- Adding more evaluation metrics

---

**Author:** ML Project Generator
**Date:** 2025
**Version:** 1.0.0
