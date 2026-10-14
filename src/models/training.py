import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Optional, Dict
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

def walk_forward_validation(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    base_model: Any,
    initial_train_size: int = 5000,
    step_size: int = 1000,
    mask_prob: float = 0.0,
    sample_weight_col: Optional[str] = None,
    calibration_method: str = 'isotonic',
    calibration_cv: int = 3
) -> Tuple[pd.DataFrame, Any]:
    """
    Performs walk-forward validation for a classification model.
    
    Args:
        df: Input DataFrame sorted by date.
        features: List of feature names.
        target: Name of the target variable.
        base_model: The base classifier to use.
        initial_train_size: Number of initial samples for training.
        step_size: Number of samples to predict in each step.
        mask_prob: Probability of masking features during training (Input Dropout).
        sample_weight_col: Column name for sample weights.
        calibration_method: Method for calibration ('isotonic' or 'sigmoid').
        calibration_cv: Number of cross-validation folds for calibration.
        
    Returns:
        Tuple containing the results DataFrame and the final trained model.
    """
    all_predictions = []
    all_indices = []
    
    model = None
    
    for current_pos in tqdm(range(initial_train_size, len(df), step_size)):
        # 1. Define training set
        X_train_raw = df[features].iloc[:current_pos]
        y_train = df[target].iloc[:current_pos]
        
        # 2. Apply Random Masking if requested
        if mask_prob > 0:
            X_train = X_train_raw.copy()
            mask = np.random.rand(*X_train.shape) < mask_prob
            X_train[mask] = np.nan
        else:
            X_train = X_train_raw
            
        # 3. Define test set
        end_pos = min(current_pos + step_size, len(df))
        X_test = df[features].iloc[current_pos:end_pos]
        
        if len(X_test) == 0:
            break
            
        # 4. Train model with Calibration
        model = CalibratedClassifierCV(base_model, method=calibration_method, cv=calibration_cv)
        
        weights = None
        if sample_weight_col and sample_weight_col in df.columns:
            weights = df[sample_weight_col].iloc[:current_pos].values
            
        model.fit(X_train, y_train, sample_weight=weights)
        
        # 5. Predict
        probs = model.predict_proba(X_test)[:, 1]
        all_predictions.extend(probs)
        all_indices.extend(df.index[current_pos:end_pos])
        
    # Create a results dataframe for the predicted portion
    results_df = df.loc[all_indices].copy()
    results_df['meta_prob'] = all_predictions
    
    return results_df, model

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, label: str = "Model") -> Dict[str, float]:
    """
    Evaluates predictions using common metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        label: Label for the model being evaluated.
        
    Returns:
        Dictionary of metrics.
    """
    ll = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    
    metrics = {
        "log_loss": ll,
        "auc": auc,
        "accuracy": acc
    }
    
    print(f"\n=== {label} Evaluation ===")
    print(f"Matches evaluated: {len(y_true)}")
    print(f"Log Loss:         {ll:.5f}")
    print(f"AUC:              {auc:.5f}")
    print(f"Accuracy:         {acc:.5f}")
    
    return metrics
