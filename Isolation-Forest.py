import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from sklearn.ensemble import IsolationForest

# Data generation
def generate_data_stream(length=1000, seasonal_period=50, anomaly_prob=0.01):
    time = np.arange(length)
    trend = time * 0.0005
    seasonal = 2 * np.sin(2 * np.pi * time / seasonal_period)
    noise = np.random.normal(0, 0.5, size=length)
    data = trend + seasonal + noise

    # Inject anomalies
    anomalies = np.random.choice([0, 1], size=length, p=[1 - anomaly_prob, anomaly_prob])
    anomaly_indices = np.where(anomalies == 1)[0]
    data[anomaly_indices] += np.random.choice([-10, 10], size=len(anomaly_indices))

    # Save to CSV
    df = pd.DataFrame({'value': data})
    df.to_csv('data_stream.csv', index=False)

    return data, anomaly_indices

# Generate and load data
data, true_anomaly_indices = generate_data_stream()
df = pd.read_csv('data_stream.csv')

# Parameters
WINDOW_SIZE = 50
CONTAMINATION = 0.05  # Proportion of anomalies in the dataset
N_ESTIMATORS = 100

# Initialize Isolation Forest model with contamination parameter
isolation_forest = IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION)

# Fit the model with the initial window of data
initial_data = np.array(df['value'].iloc[:WINDOW_SIZE]).reshape(-1, 1)
isolation_forest.fit(initial_data)

# Initialize variables for real-time anomaly detection
data_queue = deque(maxlen=WINDOW_SIZE)
scores = []
anomalies = []
xs = []
ys = []
detected_anomaly_indices = []

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

def animate(i):
    if i < len(df):
        new_value = df['value'].iloc[i]
        data_queue.append(new_value)
        xs.append(i)
        ys.append(new_value)

        # Check if the window is filled
        if len(data_queue) == WINDOW_SIZE:
            window_data = np.array(data_queue).reshape(-1, 1)
            # Predict anomaly scores for the window
            score = -isolation_forest.decision_function(window_data[-1:])
            scores.append(score[0])

            # Anomaly detection based on the prediction
            is_anomaly = isolation_forest.predict(window_data[-1:])[0] == -1
            anomalies.append(is_anomaly)

            if is_anomaly:
                detected_anomaly_indices.append(i)

            # Update the Isolation Forest model with the new data
            isolation_forest.fit(window_data)
        else:
            scores.append(np.nan)
            anomalies.append(False)

        # Update plots
        ax1.clear()
        ax2.clear()

        # Plot data stream
        ax1.plot(xs, ys, label='Data Stream')

        # Plot detected anomalies
        current_detected_anomalies = [idx for idx, val in enumerate(anomalies) if val]
        current_detected_anomaly_values = [ys[idx] for idx in current_detected_anomalies]
        ax1.scatter([xs[idx] for idx in current_detected_anomalies], current_detected_anomaly_values, color='r', label='Detected Anomalies')

        # Plot true anomalies
        current_true_anomalies = [idx for idx in true_anomaly_indices if idx < len(xs)]
        current_true_anomaly_values = [ys[idx] for idx in current_true_anomalies]
        ax1.scatter(current_true_anomalies, current_true_anomaly_values, color='g', marker='x', label='True Anomalies')

        ax1.legend()
        ax1.set_title('Real-Time Data Stream with Anomalies')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')

        # Plot anomaly scores
        ax2.plot(xs, scores, label='Anomaly Score')
        ax2.legend()
        ax2.set_title('Anomaly Score Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Anomaly Score')

        plt.tight_layout()
    else:
        ani.event_source.stop()

ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()

# Calculate performance metrics
true_positives = len(set(detected_anomaly_indices) & set(true_anomaly_indices))
false_positives = len(set(detected_anomaly_indices) - set(true_anomaly_indices))
false_negatives = len(set(true_anomaly_indices) - set(detected_anomaly_indices))
true_negatives = len(df) - (true_positives + false_positives + false_negatives)

# Precision, Recall, FPR, FNR, TPR, TNR, Accuracy, F1
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
tpr = recall
tnr = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
accuracy = (true_positives + true_negatives) / len(df)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print metrics
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"True Negatives: {true_negatives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")
print(f"TPR: {tpr:.4f}")
print(f"TNR: {tnr:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")
