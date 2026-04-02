"""
Model Training and Evaluation Script
This script loads the ice cream dataset, trains a Linear Regression model,
evaluates it, and saves the trained model for later use.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_explore_data(filepath):
    """Load the dataset and display basic information."""
    print("=" * 60)
    print("LOADING AND EXPLORING DATA")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Display basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """
    Preprocess the data:
    - Handle missing values
    - Extract features (Temperature) and target (IceCreamsSold)
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Check for missing values and drop if any
    if df.isnull().sum().sum() > 0:
        print("\nMissing values found. Dropping rows with missing values...")
        df = df.dropna()
        print(f"Dataset shape after removing nulls: {df.shape}")
    else:
        print("\nNo missing values found.")
    
    # Extract features (X) and target (y)
    X = df[['Temperature']].values
    y = df['IceCreamsSold'].values
    
    print(f"\nFeature (Temperature) shape: {X.shape}")
    print(f"Target (IceCreamsSold) shape: {y.shape}")
    
    # Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples (80%)")
    print(f"Testing set size: {X_test.shape[0]} samples (20%)")
    
    return X_train, X_test, y_train, y_test, X, y

def train_model(X_train, y_train):
    """Train the Linear Regression model."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Display model parameters
    print(f"\nModel Coefficients: {model.coef_[0]:.4f}")
    print(f"Model Intercept: {model.intercept_:.4f}")
    print(f"\nLinear Regression Equation:")
    print(f"IceCreamsSold = {model.intercept_:.4f} + {model.coef_[0]:.4f} * Temperature")
    
    return model

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Evaluate the model on both training and testing sets."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Display results
    print(f"\nTraining Set Metrics:")
    print(f"  Mean Squared Error (MSE): {train_mse:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    print(f"\nTesting Set Metrics:")
    print(f"  Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"  R² Score: {test_r2:.4f}")
    
    return y_test_pred, y_train_pred

def plot_regression(X, y, model):
    """Plot the regression line and actual data points."""
    print("\n" + "=" * 60)
    print("GENERATING REGRESSION PLOT")
    print("=" * 60)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6, s=50)
    
    # Create regression line
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_range = model.predict(X_range)
    plt.plot(X_range, y_pred_range, color='red', linewidth=2, label='Regression Line')
    
    plt.xlabel('Temperature (°F)', fontsize=12)
    plt.ylabel('Ice Creams Sold', fontsize=12)
    plt.title('Ice Cream Sales vs Temperature', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = 'static/regression_plot.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

def save_model(model, filepath='model.pkl'):
    """Save the trained model using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {filepath}")

def main():
    """Main function to orchestrate the entire training pipeline."""
    # Load and explore data
    df = load_and_explore_data('ice-cream.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X, y = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Create static directory if it doesn't exist
    import os
    os.makedirs('static', exist_ok=True)
    
    # Plot regression
    plot_regression(X, y, model)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nThe trained model is ready to use in the Flask app.")
    print("Run 'python app.py' to start the web server.")

if __name__ == '__main__':
    main()
