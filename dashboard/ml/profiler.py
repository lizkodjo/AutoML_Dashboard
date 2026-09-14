from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.services.json_utils import to_jsonable


class DataProfiler:
    """Automatically analyse datasets and generate profiles"""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialise profiler with a pandas DataFrame"""
        self.df = df
        self.profile: dict = {}

    def generate_profile(self) -> dict:
        """Generate a complete data profile."""
        self.profile = {
            "basic_info": self._get_basic_info(),
            "column_profiles": self._get_column_profiles(),
            "missing_values": self._get_missing_values(),
            "correlations": self._get_correlations(),
            "data_quality": self._get_data_quality(),
            "suggested_visualisations": self._suggest_visualisations(),
        }
        return to_jsonable(self.profile)

    def _get_basic_info(self) -> dict:
        """Return dataset-level info."""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2,
            "column_names": self.df.columns.tolist(),
            "data_types": self.df.dtypes.astype(str).to_dict(),
        }

    def _get_column_profiles(self) -> dict:
        """Generate per-column statistics."""
        profiles: dict = {}

        for col in self.df.columns:
            col_data = self.df[col]
            col_profile: dict = {
                "data_type": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
            }

            if pd.api.types.is_numeric_dtype(col_data):
                all_null = col_data.isnull().all()
                col_profile.update(
                    {
                        "mean": None if all_null else float(col_data.mean()),
                        "median": None if all_null else float(col_data.median()),
                        "std": None if all_null else float(col_data.std()),
                        "min": None if all_null else float(col_data.min()),
                        "max": None if all_null else float(col_data.max()),
                        "q1": None if all_null else float(col_data.quantile(0.25)),
                        "q3": None if all_null else float(col_data.quantile(0.75)),
                        "skewness": None if all_null else float(col_data.skew()),
                        "kurtosis": None if all_null else float(col_data.kurtosis()),
                    }
                )
                col_profile["is_target_candidate"] = self._is_target_candidate(col_data)
            else:
                col_profile.update(
                    {
                        "top_values": col_data.value_counts().head(5).to_dict(),
                        "cardinality": col_data.nunique(),
                        "high_cardinality": col_data.nunique() > 50,
                    }
                )

            profiles[col] = col_profile
        return profiles

    def _is_target_candidate(self, column: pd.Series) -> bool:
        """Return True if a numeric column could plausibly be a target."""
        if column.nunique() < 3:
            return False
        if column.std() == 0:
            return False
        return True

    def _get_missing_values(self) -> dict:
        """Summarise missing values across the dataset."""
        missing = self.df.isnull().sum()
        missing_percentage = (missing / len(self.df)) * 100

        return {
            "total_missing": missing.sum(),
            "missing_by_column": missing[missing > 0].to_dict(),
            "missing_percentage_by_column": missing_percentage[
                missing_percentage > 0
            ].to_dict(),
            "columns_with_missing": (missing > 0).sum(),
        }

    def _get_correlations(self) -> dict:
        """Return the correlation matrix for numeric columns, or {}."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            valid_cols = [col for col in numeric_cols if self.df[col].std() > 0]
            if len(valid_cols) >= 2:
                return self.df[valid_cols].corr().to_dict()
        return {}

    def _get_data_quality(self) -> dict:
        """Assess overall data quality."""
        total_cells = int(len(self.df) * len(self.df.columns))
        missing_cells = int(self.df.isnull().sum().sum())
        duplicate_rows = int(self.df.duplicated().sum())
        n_rows = max(1, len(self.df))

        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
        dup_pct = (duplicate_rows / n_rows) * 50

        quality_score = max(0.0, min(100.0, 100.0 - (missing_pct + dup_pct)))

        return {
            "quality_score": round(float(quality_score), 2),
            "duplicate_rows": duplicate_rows,
            "missing_cells": missing_cells,
            "total_cells": total_cells,
            "recommendations": self._generate_recommendations(
                missing_cells, duplicate_rows, total_cells
            ),
        }

    def _generate_recommendations(
        self, missing_cells: int, duplicate_rows: int, total_cells: int
    ) -> list[dict]:
        """Generate data-quality recommendations."""
        recommendations: list[dict] = []

        missing_percent = (missing_cells / total_cells) * 100
        if missing_percent > 5:
            recommendations.append(
                {
                    "type": "warning",
                    "message": (
                        f"High missing data: {missing_percent:.1f}% of cells are "
                        "empty. Consider imputation or removal."
                    ),
                }
            )

        if duplicate_rows > 0:
            recommendations.append(
                {
                    "type": "warning",
                    "message": (
                        f"Found {duplicate_rows} duplicate rows. Consider removing "
                        "them for cleaner analysis."
                    ),
                }
            )

        for col in self.df.columns:
            if self.df[col].dtype == "object" and self.df[col].nunique() > 100:
                recommendations.append(
                    {
                        "type": "info",
                        "message": (
                            f'Column "{col}" has {self.df[col].nunique()} unique '
                            "values. Consider feature engineering."
                        ),
                    }
                )
                break

        if not recommendations:
            recommendations.append(
                {
                    "type": "success",
                    "message": "Dataset looks clean! Ready for analysis.",
                }
            )

        return recommendations

    def _suggest_visualisations(self) -> list[dict]:
        """Suggest appropriate visualisations based on column types."""
        suggestions: list[dict] = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(
            include=["object", "category", "str"]
        ).columns

        for col in numeric_cols[:5]:
            if self.df[col].nunique() > 2:
                suggestions.append(
                    {
                        "type": "histogram",
                        "title": f"Distribution of {col}",
                        "x_column": col,
                        "distribution": "Shows the frequency distribution of values",
                    }
                )

        if len(numeric_cols) >= 3:
            suggestions.append(
                {
                    "type": "correlation_heatmap",
                    "title": "Feature Correlation Matrix",
                    "columns": numeric_cols.tolist(),
                    "description": "Shows relationships between numeric features",
                }
            )

        for col in categorical_cols[:3]:
            if self.df[col].nunique() <= 20:
                suggestions.append(
                    {
                        "type": "bar",
                        "title": f"Distribution of {col}",
                        "x_column": col,
                        "description": "Shows frequency of each category",
                    }
                )

        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols) - 1)):
                for j in range(i + 1, min(i + 2, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    suggestions.append(
                        {
                            "type": "scatter",
                            "title": f"{col1} vs {col2}",
                            "x_column": col1,
                            "y_column": col2,
                            "description": (
                                "Shows relationship between two numeric variables"
                            ),
                        }
                    )
                    break
        return suggestions[:10]
