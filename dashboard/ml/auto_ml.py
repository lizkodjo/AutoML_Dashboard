from __future__ import annotations
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


class AutoML:
    """Automated machine learning engine."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialise AutoML with a DataFrame."""
        self.df = df
        self.X = None
        self.y = None
        self.problem_type: str | None = None
        self.target_column: str | None = None
        self.models: dict = {}
        self.best_model = None
        self.best_model_name: str | None = None
        self.best_score: float = 0
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.model_results: dict = {}

    def identify_problem_type(
        self, target_col: str | None = None
    ) -> tuple[str | None, str | None]:
        """Return (problem_type, target_column).

        Auto-detects the target if none provided.
        """
        if target_col:
            self.target_column = target_col
        else:
            self.target_column = self._detect_target_column()

        if not self.target_column:
            return None, None

        target_data = self.df[self.target_column]
        if target_data.dtype == "object" or target_data.dtype.name == "category":
            self.problem_type = "classification"
        elif target_data.dtype in ["int64", "float64"]:
            self.problem_type = (
                "classification" if target_data.nunique() <= 10 else "regression"
            )
        else:
            self.problem_type = "unknown"

        return self.problem_type, self.target_column

    def _detect_target_column(self) -> str | None:
        """Automatically detect the most suitable target column."""
        scores: dict = {}

        temporal_hints = {
            "year",
            "month",
            "day",
            "week",
            "quarter",
            "date",
            "time",
            "timestamp",
            "datetime",
            "hour",
            "minute",
        }
        target_hints = {
            "target",
            "label",
            "outcome",
            "result",
            "class",
            "survived",
            "value",
            "arrivals",
            "sales",
            "revenue",
            "price",
            "amount",
            "score",
            "rating",
        }
        id_hints = {"id", "uuid", "index", "key"}

        for col in self.df.columns:
            if self.df[col].isnull().sum() / len(self.df) > 0.4:
                continue

            name = str(col).lower()
            score = 0

            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].std() > 0:
                    score += 3
                if self.df[col].nunique() > 20:
                    score += 1
            else:
                if 2 <= self.df[col].nunique() <= 20:
                    score += 3

            tokens = set(re.split(r"[^a-z0-9]+", name))
            if tokens & target_hints:
                score += 3
            if any(hint in name for hint in temporal_hints):
                score -= 10
            if any(hint in name for hint in id_hints):
                score -= 8
            if self.df[col].isnull().all():
                score -= 20

            scores[col] = score

        if not scores:
            return None

        best_col = max(scores, key=scores.get)
        return best_col if scores[best_col] > 0 else None

    def prepare_data(self, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training: impute, encode, scale, split."""
        if not self.target_column:
            self.identify_problem_type()

        all_null_cols = [
            c
            for c in self.df.columns
            if c != self.target_column and self.df[c].isnull().all()
        ]
        if all_null_cols:
            self.df = self.df.drop(columns=all_null_cols)

        feature_cols = [col for col in self.df.columns if col != self.target_column]
        X = self.df[feature_cols].copy()
        y = self.df[self.target_column].copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "str", "string"]
        ).columns.tolist()

        if numeric_cols:
            X_num = X[numeric_cols].copy()
            keep_num = [c for c in numeric_cols if not X_num[c].isnull().all()]
            dropped = set(numeric_cols) - set(keep_num)
            if dropped:
                X = X.drop(columns=list(dropped))
                numeric_cols = keep_num

            if numeric_cols:
                imputer_num = SimpleImputer(strategy="mean")
                X[numeric_cols] = pd.DataFrame(
                    imputer_num.fit_transform(X[numeric_cols]),
                    columns=numeric_cols,
                    index=X.index,
                )

        if categorical_cols:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            X[categorical_cols] = pd.DataFrame(
                imputer_cat.fit_transform(X[categorical_cols].astype(str)),
                columns=categorical_cols,
                index=X.index,
            )

        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le

        y_encoded = y.copy()
        if self.problem_type == "classification" and y.dtype == "object":
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y_encoded.astype(str))
            self.label_encoders["target"] = le_target
        elif self.problem_type == "classification" and y.nunique() <= 20:
            y_encoded = y_encoded.astype(int)
        elif self.problem_type == "regression" and y.dtype == "object":
            self.problem_type = "classification"
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y_encoded.astype(str))
            self.label_encoders["target"] = le_target

        X_scaled = self.scaler.fit_transform(X_encoded)
        self.X = X_scaled
        self.y = y_encoded
        stratify_arg = y_encoded if self.problem_type == "classification" else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled,
            y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=stratify_arg,
        )

        return X_scaled, y_encoded

    def train_models(self) -> dict:
        """Train several models and select the best performer."""
        if self.X is None:
            self.prepare_data()

        if self.problem_type == "classification":
            models_config = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42
                ),
            }
        else:
            models_config = {
                "Random Forest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
                "Linear Regression": LinearRegression(),
            }

        for name, model in models_config.items():
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                if self.problem_type == "classification":
                    score = accuracy_score(self.y_test, y_pred)
                else:
                    score = r2_score(self.y_test, y_pred)

                if hasattr(model, "feature_importances_"):
                    self.feature_importance = model.feature_importances_

                self.models[name] = model
                self.model_results[name] = {
                    "score": score,
                    "predictions": y_pred.tolist(),
                    "model_type": self.problem_type,
                }

                if score > self.best_score or self.best_model is None:
                    self.best_score = score
                    self.best_model = model
                    self.best_model_name = name
            except Exception as e:  # noqa: BLE001
                print(f"Error training {name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return self.model_results

    def get_performance_report(self) -> dict:
        """Generate a detailed performance report."""
        if not self.model_results:
            self.train_models()

        report: dict = {
            "problem_type": self.problem_type,
            "target_column": self.target_column,
            "training_samples": len(self.X_train) if hasattr(self, "X_train") else 0,
            "test_samples": len(self.X_test) if hasattr(self, "X_test") else 0,
            "feature_count": self.X.shape[1] if self.X is not None else 0,
            "model_performance": {},
            "best_model": self.best_model_name,
            "best_score": self.best_score,
        }

        for name, results in self.model_results.items():
            report["model_performance"][name] = {
                "score": results["score"],
                "metric": (
                    "accuracy" if self.problem_type == "classification" else "r2_score"
                ),
            }

        if self.feature_importance is not None:
            feature_names = [
                col for col in self.df.columns if col != self.target_column
            ]
            importance_dict = dict(zip(feature_names, self.feature_importance))
            report["feature_importance"] = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:10]

        return report

    def predict(self, new_data: pd.DataFrame) -> list:
        """Make predictions on new data using the best model."""
        if self.best_model is None:
            self.train_models()

        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if self.target_column in new_data.columns:
            new_data = new_data.drop(columns=[self.target_column])

        return self.best_model.predict(new_data).tolist()

    def save_model(self, filepath: str) -> bool:
        """Save the best model and its preprocessing bundle to disk."""
        if self.best_model is None:
            return False

        feature_cols = [c for c in self.df.columns if c != self.target_column]
        bundle = {
            "model": self.best_model,
            "best_model_name": self.best_model_name or "unknown",
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "feature_columns": feature_cols,
            "best_score": self.best_score,
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        joblib.dump(bundle, filepath)
        return True

    def load_model(self, filepath: str) -> bool:
        """Load a saved model bundle."""
        if not os.path.exists(filepath):
            return False

        bundle = joblib.load(filepath)
        self.best_model = bundle["model"]
        self.best_model_name = bundle.get("best_model_name", "loaded model")
        self.scaler = bundle["scaler"]
        self.label_encoders = bundle["label_encoders"]
        self.target_column = bundle["target_column"]
        self.problem_type = bundle["problem_type"]
        self.feature_columns = bundle["feature_columns"]
        self.best_score = bundle.get("best_score", 0.0)
        return True
