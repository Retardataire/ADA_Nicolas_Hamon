import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, mean_absolute_error
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.data_preprocessing import preprocess_data
import os
import datetime

def train_random_forest(X_train, y_train, param_grid):
    """Train Random Forest with hyperparameter search."""
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_name, save_path):
    """Evaluate model performance and save results."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print(f"{model_name} Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{save_path}{model_name}_confusion_matrix.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{save_path}{model_name}_roc_curve.png')
    plt.close()

    mae = mean_absolute_error(y_test, y_pred)
    print(f"{model_name} Mean Absolute Error (MAE): {mae:.4f}")

    return auc, mae, report

def plot_feature_importances(importances, feature_names, save_path):
    """Plot feature importances."""
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importances')
    plt.savefig(f'{save_path}feature_importances.png')
    plt.close()

    return feature_importance

def rebalance_dataset(X_train, y_train, random_state=42):
    """Rebalance the training dataset using SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

def create_results_dir(run_name):
    """Create a directory for saving results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"../results/{run_name}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name + "/"

def main():
    # Load data
    DATA_DIR = '../data'
    TRAINING_FILE = 'training_data.csv'
    df = pd.read_csv(f"{DATA_DIR}/{TRAINING_FILE}")

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training set label distribution:")
    print(pd.Series(y_train).value_counts())
    print("\nTest set label distribution:")
    print(pd.Series(y_test).value_counts())

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameter grid for Random Forest
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # --- Run 1: Train on Imbalanced Data ---
    print("\n=== Training on Imbalanced Data ===")
    imbalanced_dir = create_results_dir("rf_imbalanced")
    rf_model_imbalanced, best_params_imbalanced = train_random_forest(X_train_scaled, y_train, param_grid, "imbalanced")
    rf_auc_imbalanced, rf_mae_imbalanced, report_imb = evaluate_model(rf_model_imbalanced, X_test_scaled, y_test, 'Random Forest (Imbalanced)', imbalanced_dir)
    feature_importance_imbalanced = plot_feature_importances(rf_model_imbalanced.feature_importances_, X.columns, imbalanced_dir)

    # --- Run 2: Train on Balanced Data ---
    print("\n=== Training on Balanced Data ===")
    X_train_resampled, y_train_resampled = rebalance_dataset(X_train_scaled, y_train)
    balanced_dir = create_results_dir("rf_balanced")
    rf_model_balanced, best_params_balanced = train_random_forest(X_train_resampled, y_train_resampled, param_grid, "balanced")
    rf_auc_balanced, rf_mae_balanced, report_b = evaluate_model(rf_model_balanced, X_test_scaled, y_test, 'Random Forest (Balanced)', balanced_dir)
    feature_importance_balanced = plot_feature_importances(rf_model_balanced.feature_importances_, X.columns, balanced_dir)

    # Save results
    with open(f"{imbalanced_dir}rf_results.txt", 'w') as f:
        f.write("=== Random Forest (Imbalanced) ===\n")
        f.write(f"Best Parameters: {best_params_imbalanced}\n")
        f.write(f"AUC: {rf_auc_imbalanced:.4f}\n")
        f.write(f"MAE: {rf_mae_imbalanced:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_imb)
        f.write("\nFeature Importances:\n")
        f.write(feature_importance_imbalanced.to_string())

    with open(f"{balanced_dir}rf_results.txt", 'w') as f:
        f.write("=== Random Forest (Balanced) ===\n")
        f.write(f"Best Parameters: {best_params_balanced}\n")
        f.write(f"AUC: {rf_auc_balanced:.4f}\n")
        f.write(f"MAE: {rf_mae_balanced:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_b)
        f.write("\nFeature Importances:\n")
        f.write(feature_importance_balanced.to_string())

if __name__ == "__main__":
    main()
