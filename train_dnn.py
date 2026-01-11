import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.data_preprocessing import preprocess_data
import os
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import keras_tuner as kt

def build_model(hp, input_shape):
    """Build the DNN model with hyperparameters to tune."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Tune the number of units in the first Dense layer
    hp_units_1 = hp.Int('units_1', min_value=64, max_value=512, step=64)
    model.add(Dense(units=hp_units_1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Tune the number of units in the second Dense layer
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=256, step=32)
    model.add(Dense(units=hp_units_2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    # Tune the number of units in the third Dense layer
    hp_units_3 = hp.Int('units_3', min_value=16, max_value=128, step=16)
    model.add(Dense(units=hp_units_3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_dnn(X_train, y_train, input_shape, run_name):
    """Train DNN with hyperparameter search using KerasTuner."""
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name=run_name
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        restore_best_weights=True
    )

    tuner.search(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters found:")
    print(f"First layer units: {best_hps.get('units_1')}")
    print(f"Second layer units: {best_hps.get('units_2')}")
    print(f"Third layer units: {best_hps.get('units_3')}")
    print(f"First layer dropout: {best_hps.get('dropout_1')}")
    print(f"Second layer dropout: {best_hps.get('dropout_2')}")
    print(f"Third layer dropout: {best_hps.get('dropout_3')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")

    return best_model, best_hps

def evaluate_model(model, X_test, y_test, model_name, save_path):
    """Evaluate model performance and save results."""
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

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

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
    plt.savefig(f'{save_path}{model_name}_precision_recall_curve.png')
    plt.close()

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    return auc, accuracy, report

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

    input_shape = (X_train_scaled.shape[1],)

    # --- Run 1: Train on Imbalanced Data ---
    print("\n=== Training on Imbalanced Data ===")
    imbalanced_dir = create_results_dir("dnn_imbalanced")
    dnn_model_imbalanced, best_params_imbalanced = train_dnn(X_train_scaled, y_train, input_shape, "dnn_imbalanced")
    dnn_auc_imbalanced, dnn_accuracy_imbalanced, report_imb = evaluate_model(dnn_model_imbalanced, X_test_scaled, y_test, 'DNN (Imbalanced)', imbalanced_dir)

    # --- Run 2: Train on Balanced Data ---
    print("\n=== Training on Balanced Data ===")
    X_train_resampled, y_train_resampled = rebalance_dataset(X_train_scaled, y_train)
    balanced_dir = create_results_dir("dnn_balanced")
    dnn_model_balanced, best_params_balanced = train_dnn(X_train_resampled, y_train_resampled, input_shape, "dnn_balanced")
    dnn_auc_balanced, dnn_accuracy_balanced, report_b = evaluate_model(dnn_model_balanced, X_test_scaled, y_test, 'DNN (Balanced)', balanced_dir)

    # Save results
    with open(f"{imbalanced_dir}dnn_results.txt", 'w') as f:
        f.write("=== DNN (Imbalanced) ===\n")
        f.write(f"Best Parameters: {best_params_imbalanced.values}\n")
        f.write(f"AUC: {dnn_auc_imbalanced:.4f}\n")
        f.write(f"Accuracy: {dnn_accuracy_imbalanced:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_imb)

    with open(f"{balanced_dir}dnn_results.txt", 'w') as f:
        f.write("=== DNN (Balanced) ===\n")
        f.write(f"Best Parameters: {best_params_balanced.values}\n")
        f.write(f"AUC: {dnn_auc_balanced:.4f}\n")
        f.write(f"Accuracy: {dnn_accuracy_balanced:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_b)

if __name__ == "__main__":
    main()
