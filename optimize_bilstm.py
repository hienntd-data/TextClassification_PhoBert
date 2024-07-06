import optuna
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from optuna.integration import TFKerasPruningCallback
import pickle
from optuna.visualization import plot_optimization_history
import optuna.visualization as ov
from optuna.trial import TrialState

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""### **Load data**"""

# Load data
with open('data/features_162k_phobertbase.pkl', 'rb') as f:
    data_dict = pickle.load(f)


X_train = np.array(data_dict['X_train'])
X_val = np.array(data_dict['X_val'])
X_test = np.array(data_dict['X_test'])
y_train = data_dict['y_train']
y_val = data_dict['y_val']
y_test = data_dict['y_test']

y_train = y_train.values.astype(int)
y_test = y_test.values.astype(int)
y_val = y_val.values.astype(int)

"""##**Build Model**"""

# Define the BiLSTM model architecture
def build_bilstm_model(lstm_units_1, lstm_units_2, dense_units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    # LSTM Layer 1 with dropout
    model.add(Bidirectional(LSTM(lstm_units_1, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    # LSTM Layer 2 with dropout
    model.add(Bidirectional(LSTM(lstm_units_2, return_sequences=False)))
    model.add(Dropout(dropout_rate))
    # Dense Layer with dropout and ReLU activation
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    # Final Dense Layer with softmax activation
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # Use Adam optimizer with the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

"""##**Create objective**"""

# Define the objective function for optimization
def objective_bilstm(trial):
    lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 512, step=32)
    lstm_units_2 = trial.suggest_int('lstm_units_2', lstm_units_1//2, lstm_units_1, step=32)
    dense_units = trial.suggest_int('dense_units', 64, 512, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 10, 30, step=10)
    batch_size = trial.suggest_int('batch_size', 64, 256, step=32)

    print(f"Trying hyperparameters: lstm_units_1={lstm_units_1}, lstm_units_2={lstm_units_2}, dense_units={dense_units}, "
          f"dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}")

    model = build_bilstm_model(lstm_units_1, lstm_units_2, dense_units, dropout_rate, learning_rate)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=[TFKerasPruningCallback(trial, "val_loss")], verbose=1)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return accuracy

"""##**Study to find hyperparameters**"""

# Create an Optuna study for optimization
study_bilstm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
study_bilstm.optimize(lambda trial: objective_bilstm(trial), n_trials=100)

# Save completed trials to a CSV file
complete_trials = study_bilstm.trials_dataframe()[study_bilstm.trials_dataframe()['state'] == 'COMPLETE']
complete_trials.to_csv("assets/study_bilstm_256_trials.csv", index=False)

# Extract the best hyperparameters
best_hyperparameters_bilstm = study_bilstm.best_trial.params

# Save the best hyperparameters to a JSON file
with open('hyperparameters/BiLSTM_phobertbase.json', 'w') as file:
    json.dump(best_hyperparameters_bilstm, file)

plot_optimization_history(study_bilstm)

html_file_path = "images/study_bilstm_phobertbase_optimize_history.html"
# Plot and save the optimization history plot as an HTML file
ov.plot_optimization_history(study_bilstm).write_html(html_file_path)
plot_optimization_history(study_bilstm)
