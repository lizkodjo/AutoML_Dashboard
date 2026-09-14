from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.services.json_utils import to_jsonable


class Visualiser:
    """Automatically generates various chart types based on data."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialise visualiser with a DataFrame."""
        self.df = df
        self.charts: dict = {}

    @staticmethod
    def _is_id_like(col_name: str) -> bool:
        """Detect ID-like column names."""
        name = str(col_name).lower()
        id_hints = ["id", "uuid", "index"]
        return (
            any(hint in name for hint in id_hints)
            and "valid" not in name
            and "grid" not in name
        )

    def generate_chart(self, chart_config: dict) -> str:
        """Generate a chart from a configuration dict; return Plotly JSON."""
        chart_type = chart_config.get("type")
        title = chart_config.get("title", "")

        if chart_type == "histogram":
            fig = self._create_histogram(chart_config)
        elif chart_type == "bar":
            fig = self._create_bar_chart(chart_config)
        elif chart_type == "scatter":
            fig = self._create_scatter_plot(chart_config)
        elif chart_type == "correlation_heatmap":
            fig = self._create_heatmap(chart_config)
        elif chart_type == "line":
            fig = self._create_line_chart(chart_config)
        elif chart_type == "box":
            fig = self._create_box_plot(chart_config)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        return fig.to_json()

    def _create_histogram(self, config: dict) -> go.Figure:
        """Create a histogram."""
        x_col = config.get("x_column")
        data = self.df[x_col].dropna()
        if len(data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data in {x_col}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        counts, bin_edges = np.histogram(data, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=bin_centers.tolist(),
                    y=counts.tolist(),
                    width=bin_widths.tolist(),
                    name=str(x_col),
                    hovertemplate=f"{x_col}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                )
            ]
        )
        fig.update_layout(xaxis_title=str(x_col), yaxis_title="Count", bargap=0.05)
        return fig

    def _create_bar_chart(self, config: dict) -> go.Figure:
        """Create a bar chart."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")

        if y_col:
            return px.bar(
                self.df,
                x=x_col,
                y=y_col,
                title=config.get("title", f"{y_col} by {x_col}"),
            )
        value_counts = self.df[x_col].value_counts().reset_index()
        value_counts.columns = [x_col, "count"]
        value_counts[x_col] = value_counts[x_col].astype(str)
        value_counts["count"] = value_counts["count"].astype(int)
        return px.bar(
            value_counts,
            x=x_col,
            y="count",
            title=config.get("title", f"Count of {x_col}"),
        )

    def _create_scatter_plot(self, config: dict) -> go.Figure:
        """Create a scatter plot."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        color_col = config.get("color_column")

        if not y_col:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            candidates = [c for c in numeric_cols if c != x_col]
            y_col = candidates[0] if candidates else None

        if x_col not in self.df.columns or y_col not in self.df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Invalid columns for scatter plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        cols_needed = [x_col, y_col]
        has_color = (
            color_col
            and color_col in self.df.columns
            and color_col != x_col
            and color_col != y_col
        )
        if has_color:
            cols_needed.append(color_col)

        plot_df = self.df[cols_needed].dropna(subset=[x_col, y_col]).copy()

        if len(plot_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No valid data for {x_col} vs {y_col}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        x_data = plot_df[x_col].astype(float).tolist()
        y_data = plot_df[y_col].astype(float).tolist()

        if has_color:
            fig = go.Figure()
            for cat_value, group in plot_df.groupby(color_col, dropna=True):
                fig.add_trace(
                    go.Scattergl(
                        x=group[x_col].astype(float).tolist(),
                        y=group[y_col].astype(float).tolist(),
                        mode="markers",
                        name=str(cat_value),
                        marker=dict(size=7, opacity=0.7),
                        hovertemplate=(
                            f"{x_col}: %{{x}}<br>{y_col}: %{{y}}"
                            f"<extra>{cat_value}</extra>"
                        ),
                    )
                )
        else:
            fig = go.Figure(
                data=go.Scattergl(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    marker=dict(size=7, opacity=0.6, color="#667eea"),
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
                )
            )

        fig.update_layout(xaxis_title=str(x_col), yaxis_title=str(y_col))
        return fig

    def _create_heatmap(self, config: dict) -> go.Figure:
        """Create a correlation heatmap."""
        columns = config.get("columns")
        if not columns:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough numeric columns for correlation heatmap",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        corr_matrix = self.df[columns].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values.tolist(),
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"))
        return fig

    def _create_line_chart(self, config: dict) -> go.Figure:
        """Create a line chart."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")

        if not y_col:
            numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
            y_col = (
                [col for col in numeric_cols if col != x_col][0]
                if len(numeric_cols) > 1
                else None
            )

        return px.line(
            self.df,
            x=x_col,
            y=y_col,
            title=config.get("title", f"{y_col} over {x_col}"),
        )

    def _create_box_plot(self, config: dict) -> go.Figure:
        """Create a box plot."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        return px.box(
            self.df,
            x=x_col,
            y=y_col,
            title=config.get("title", f"Distribution of {y_col} by {x_col}"),
        )

    def generate_auto_charts(self, num_charts: int = 5) -> dict:
        """Auto-generate a set of charts based on column types."""
        charts: dict = {}

        numeric_cols = [
            c
            for c in self.df.select_dtypes(include=[np.number]).columns
            if not self._is_id_like(c)
        ]
        categorical_cols = [
            c for c in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[c])
        ]

        chart_count = 0

        # 1. Histograms for numeric columns
        for col in numeric_cols[:3]:
            if chart_count >= num_charts:
                break
            if self.df[col].nunique() > 2:
                config = {
                    "type": "histogram",
                    "x_column": str(col),
                    "title": f"Distribution of {col}",
                }
                charts[f"hist_{col}"] = {
                    "config": to_jsonable(config),
                    "figure": self.generate_chart(config),
                }
                chart_count += 1

        # 2. Bar charts for categorical columns (+ low-cardinality numerics)
        cat_like_cols: list = list(categorical_cols)
        for c in numeric_cols:
            if self.df[c].nunique() <= 5:
                cat_like_cols.append(c)

        for col in cat_like_cols[:3]:
            if chart_count >= num_charts:
                break
            if self.df[col].nunique() <= 20:
                config = {
                    "type": "bar",
                    "x_column": str(col),
                    "title": f"Distribution of {col}",
                }
                charts[f"bar_{col}"] = {
                    "config": to_jsonable(config),
                    "figure": self.generate_chart(config),
                }
                chart_count += 1

        # 3. Correlation heatmap
        if len(numeric_cols) >= 3 and chart_count < num_charts:
            config = {
                "type": "correlation_heatmap",
                "columns": [str(c) for c in numeric_cols],
                "title": "Correlation Matrix",
            }
            charts["heatmap"] = {
                "config": to_jsonable(config),
                "figure": self.generate_chart(config),
            }
            chart_count += 1

        # 4. Scatter plots for continuous columns
        continuous_cols = [c for c in numeric_cols if self.df[c].nunique() > 10]
        if len(continuous_cols) >= 2 and chart_count < num_charts:
            for i in range(min(2, len(continuous_cols) - 1)):
                if chart_count >= num_charts:
                    break
                x_col = continuous_cols[i]
                y_col = continuous_cols[i + 1]
                config = {
                    "type": "scatter",
                    "x_column": str(x_col),
                    "y_column": str(y_col),
                    "title": f"{x_col} vs {y_col}",
                }
                charts[f"scatter_{x_col}_{y_col}"] = {
                    "config": to_jsonable(config),
                    "figure": self.generate_chart(config),
                }
                chart_count += 1

        return charts
