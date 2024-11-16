import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict

class HousePricePipeline:
    def __init__(self):
        self.preprocessing_pipeline = None
        self.model = None
        self.feature_names = [
            'Avg. Area Income',
            'Avg. Area House Age',
            'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms',
            'Area Population'
        ]

    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create the preprocessing pipeline."""
        numeric_features = self.feature_names

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

        return Pipeline([
            ('preprocessor', preprocessor)
        ])

    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the dataset."""
        X = X.copy()
        # Create total rooms to bedrooms ratio
        X['Rooms_to_Bedrooms_Ratio'] = X['Avg. Area Number of Rooms'] / X['Avg. Area Number of Bedrooms']
        self.feature_names.append('Rooms_to_Bedrooms_Ratio')
        return X

    def create_ensemble_model(self):
        """Create an ensemble of multiple regression models."""
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        
        return models

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train multiple regression models."""
        trained_models = {}
        models = self.create_ensemble_model()
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models

    def ensemble_predictions(self, models: Dict, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble of models."""
        predictions = np.column_stack([
            model.predict(X) for model in models.values()
        ])
        return np.mean(predictions, axis=1)

    def train(self, data_path: str) -> Tuple[Dict[str, float], Dict]:
        """Train the complete pipeline and return metrics."""
        # Load and prepare data
        df = pd.read_csv(data_path)
        df = df.drop('Address', axis=1)
        
        # Feature engineering
        X = self.feature_engineering(df.drop('Price', axis=1))
        y = df['Price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and fit preprocessing pipeline
        self.preprocessing_pipeline = self.create_preprocessing_pipeline()
        X_train_processed = self.preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = self.preprocessing_pipeline.transform(X_test)
        
        # Train models
        trained_models = self.train_models(X_train_processed, y_train)
        
        # Make predictions
        train_predictions = self.ensemble_predictions(trained_models, X_train_processed)
        test_predictions = self.ensemble_predictions(trained_models, X_test_processed)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, train_predictions),
            'test_r2': r2_score(y_test, test_predictions),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_predictions))
        }
        
        # Save pipelines
        joblib.dump(self.preprocessing_pipeline, 'preprocessing_pipeline.pkl')
        joblib.dump(trained_models, 'trained_models.pkl')
        
        return metrics, trained_models

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        # Feature engineering
        X = self.feature_engineering(X)
        
        # Preprocess the data
        X_processed = self.preprocessing_pipeline.transform(X)
        
        # Load models
        trained_models = joblib.load('trained_models.pkl')
        
        # Make predictions
        predictions = self.ensemble_predictions(trained_models, X_processed)
        return predictions