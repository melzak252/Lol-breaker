import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from typing import List, Dict, Optional, Union

def plot_calibration_curve(
    results: List[Dict[str, Union[pd.Series, str]]],
    n_bins: int = 10,
    strategy: str = 'uniform',
    title: str = "Calibration Curve (Reliability Diagram)",
    save_path: Optional[str] = None
):
    """
    Plots calibration curves for one or more models.
    
    Args:
        results: List of dictionaries, each containing:
            - 'y_true': True labels (Series or array)
            - 'y_prob': Predicted probabilities (Series or array)
            - 'label': Label for the model
            - 'color': (Optional) Color for the plot
        n_bins: Number of bins for calibration curve.
        strategy: Strategy for binning ('uniform' or 'quantile').
        title: Title of the plot.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for res in results:
        y_true = res['y_true']
        y_prob = res['y_prob']
        label = res['label']
        color = res.get('color')
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )
        
        plt.plot(
            mean_predicted_value, 
            fraction_of_positives, 
            "s-", 
            label=f"{label} (n={len(y_true)})",
            color=color
        )
        
    plt.ylabel("Fraction of positives (Actual Win Rate)")
    plt.xlabel("Mean predicted value (Predicted Probability)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    method: str = 'spearman',
    save_path: Optional[str] = None,
    vmin: float = 0.8,
    vmax: float = 1.0
):
    """
    Plots a correlation heatmap for a DataFrame.
    """
    corr_matrix = df.corr(method=method)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved correlation heatmap to {save_path}")
    else:
        plt.show()
    
    return corr_matrix
