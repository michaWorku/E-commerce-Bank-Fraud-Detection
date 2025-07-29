import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a regression model using RMSE and R-squared.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        Dict[str, float]: A dictionary containing RMSE and R-squared.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Empty true or predicted arrays for regression evaluation.")
        return {'RMSE': np.nan, 'R-squared': np.nan}
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r_squared = r2_score(y_true, y_pred)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    
    return {'RMSE': rmse, 'R-squared': r_squared}

def evaluate_classification_model(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluates a binary classification model using accuracy, precision, recall, F1-score, and ROC-AUC.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        threshold (float): Probability threshold for converting probabilities to binary predictions.

    Returns:
        Dict[str, float]: A dictionary containing accuracy, precision, recall, F1-score, and ROC-AUC.
    """
    if len(y_true) == 0 or len(y_pred_proba) == 0:
        print("Warning: Empty true or predicted arrays for classification evaluation.")
        return {'Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'F1-score': np.nan, 'ROC-AUC': np.nan}
        
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0) # Handle cases where no positive predictions
    recall = recall_score(y_true, y_pred_binary, zero_division=0) # Handle cases where no actual positives
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC-AUC requires probabilities
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        # This can happen if y_true contains only one class, for example
        roc_auc = np.nan
        print("Warning: ROC-AUC could not be calculated (e.g., only one class present in y_true).")

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'ROC-AUC': roc_auc
    }
