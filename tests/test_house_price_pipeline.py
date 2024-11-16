import pytest
import pandas as pd
import numpy as np
from housereg.house_price_pipeline import HousePricePipeline
from sklearn.pipeline import Pipeline
import joblib

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Avg. Area Income': [79545.45857, 81005.21473],
        'Avg. Area House Age': [5.682861322, 7.009188142],
        'Avg. Area Number of Rooms': [7.009188142, 6.730821019],
        'Avg. Area Number of Bedrooms': [3.09, 3.09],
        'Area Population': [23086.80005, 40173.07217],
        'Price': [1059033.558, 1505890.876]
    })

@pytest.fixture
def pipeline():
    """Create a pipeline instance for testing."""
    return HousePricePipeline()

def test_create_preprocessing_pipeline(pipeline):
    """Test if preprocessing pipeline is created correctly."""
    preprocessing_pipeline = pipeline.create_preprocessing_pipeline()
    assert isinstance(preprocessing_pipeline, Pipeline)
    assert 'preprocessor' in preprocessing_pipeline.named_steps

def test_feature_engineering(pipeline, sample_data):
    """Test feature engineering functionality."""
    X = sample_data.drop('Price', axis=1)
    engineered_data = pipeline.feature_engineering(X)
    
    assert 'Rooms_to_Bedrooms_Ratio' in engineered_data.columns
    assert len(engineered_data.columns) == len(X.columns) + 1
    assert all(engineered_data['Rooms_to_Bedrooms_Ratio'] == 
              engineered_data['Avg. Area Number of Rooms'] / 
              engineered_data['Avg. Area Number of Bedrooms'])

def test_create_ensemble_model(pipeline):
    """Test ensemble model creation."""
    models = pipeline.create_ensemble_model()
    assert len(models) == 3
    assert all(key in models for key in ['rf', 'gb', 'lr'])

def test_train_models(pipeline, sample_data):
    """Test model training functionality."""
    X = sample_data.drop('Price', axis=1)
    y = sample_data['Price']
    
    # Preprocess data
    preprocessing_pipeline = pipeline.create_preprocessing_pipeline()
    X_processed = preprocessing_pipeline.fit_transform(X)
    
    # Train models
    trained_models = pipeline.train_models(X_processed, y)
    
    assert len(trained_models) == 3
    assert all(hasattr(model, 'predict') for model in trained_models.values())

def test_ensemble_predictions(pipeline, sample_data):
    """Test ensemble predictions."""
    X = sample_data.drop('Price', axis=1)
    y = sample_data['Price']
    
    # Preprocess data
    preprocessing_pipeline = pipeline.create_preprocessing_pipeline()
    X_processed = preprocessing_pipeline.fit_transform(X)
    
    # Train models
    trained_models = pipeline.train_models(X_processed, y)
    
    # Make predictions
    predictions = pipeline.ensemble_predictions(trained_models, X_processed)
    
    assert len(predictions) == len(X)
    assert isinstance(predictions, np.ndarray)
    assert not np.any(np.isnan(predictions))

