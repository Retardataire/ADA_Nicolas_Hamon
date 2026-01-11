import os
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scripts.data_preprocessing import preprocess_data
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score, precision_recall_curve, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import keras_tuner as kt

def build_model(hp, input_shape):
    """Build the LSTM model with hyperparameters to tune."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Tune the number of units in the first LSTM layer
    hp_lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=256, step=32)
    model.add(Bidirectional(LSTM(units=hp_lstm_units_1, return_sequences=True)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Tune the number of units in the second LSTM layer
    hp_lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=128, step=16)
    model.add(Bidirectional(LSTM(units=hp_lstm_units_2)))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

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

def train_lstm(X_train, y_train, input_shape, run_name):
    """Train LSTM with hyperparameter search using KerasTuner."""
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective=kt.Objective("accuracy", direction="max"),
        max_trials=10,
        executions_per_trial=2,
        directory='keras_tuner_dir',
        project_name=run_name
    )

    early_stopping = EarlyStopping(
        monitor='accuracy',
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
    print(f"First LSTM layer units: {best_hps.get('lstm_units_1')}")
    print(f"Second LSTM layer units: {best_hps.get('lstm_units_2')}")
    print(f"First layer dropout: {best_hps.get('dropout_1')}")
    print(f"Second layer dropout: {best_hps.get('dropout_2')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")

    return best_model, best_hps

def reshape_for_lstm(X):
    """Reshape data for LSTM input (samples, timesteps, features)."""
    return X.reshape((X.shape[0], 1, X.shape[1]))

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

    # F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} F1-score: {f1:.4f}")

    return auc, f1, report

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

    # Reshape data for LSTM
    X_train_reshaped = reshape_for_lstm(X_train_scaled)
    X_test_reshaped = reshape_for_lstm(X_test_scaled)

    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])

    # Calculate class weights
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weights = dict(enumerate(class_weights))
    # print("Class weights:", class_weights)

    # --- Run 1: Train on Imbalanced Data ---
    print("\n=== Training on Imbalanced Data ===")
    imbalanced_dir = create_results_dir("lstm_imbalanced")
    lstm_model_imbalanced, best_params_imbalanced = train_lstm(X_train_reshaped, y_train, input_shape, "lstm_imbalanced")
    lstm_auc_imbalanced, lstm_f1_imbalanced, report_imb = evaluate_model(lstm_model_imbalanced, X_test_reshaped, y_test, 'LSTM (Imbalanced)', imbalanced_dir)

    # --- Run 2: Train on Balanced Data ---
    print("\n=== Training on Balanced Data ===")
    X_train_resampled, y_train_resampled = rebalance_dataset(X_train_scaled, y_train)
    X_train_resampled_reshaped = reshape_for_lstm(X_train_resampled)
    balanced_dir = create_results_dir("lstm_balanced")
    lstm_model_balanced, best_params_balanced = train_lstm(X_train_resampled_reshaped, y_train_resampled, input_shape, "lstm_balanced")
    lstm_auc_balanced, lstm_f1_balanced, report_b = evaluate_model(lstm_model_balanced, X_test_reshaped, y_test, 'LSTM (Balanced)', balanced_dir)

    # Save results
    with open(f"{imbalanced_dir}lstm_results.txt", 'w') as f:
        f.write("=== LSTM (Imbalanced) ===\n")
        f.write(f"Best Parameters: {best_params_imbalanced.values}\n")
        f.write(f"AUC: {lstm_auc_imbalanced:.4f}\n")
        f.write(f"F1-score: {lstm_f1_imbalanced:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_imb)

    with open(f"{balanced_dir}lstm_results.txt", 'w') as f:
        f.write("=== LSTM (Balanced) ===\n")
        f.write(f"Best Parameters: {best_params_balanced.values}\n")
        f.write(f"AUC: {lstm_auc_balanced:.4f}\n")
        f.write(f"F1-score: {lstm_f1_balanced:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_b)

if __name__ == "__main__":
    main()
