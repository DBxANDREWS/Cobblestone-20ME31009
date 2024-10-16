import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

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
THRESHOLD = 2.5  # Adjust as needed

# Initialize variables
data_queue = deque(maxlen=WINDOW_SIZE)
anomalies = []
xs = []
ys = []
detected_anomaly_indices = []

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1)

def animate(i):
    if i < len(df):
        new_value = df['value'].iloc[i]
        data_queue.append(new_value)
        xs.append(i)
        ys.append(new_value)

        if len(data_queue) == WINDOW_SIZE:
            # Get data for Z-score calculation
            window = list(data_queue)
            mean = np.mean(window)
            std = np.std(window)
            if std == 0:
                std = 1e-6  # Avoid division by zero

            z_score = abs((new_value - mean) / std)
            is_anomaly = z_score > THRESHOLD
            anomalies.append(is_anomaly)

            if is_anomaly:
                detected_anomaly_indices.append(i)
        else:
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

        # Plot Z-score
        z_scores = []
        for idx in range(len(anomalies)):
            if idx >= WINDOW_SIZE - 1:
                window = ys[idx - WINDOW_SIZE + 1:idx + 1]
                mean = np.mean(window)
                std = np.std(window)
                if std == 0:
                    std = 1e-6
                z = abs((ys[idx] - mean) / std)
                z_scores.append(z)
            else:
                z_scores.append(np.nan)

        ax2.plot(xs, z_scores, label='Z-score')
        ax2.axhline(y=THRESHOLD, color='r', linestyle='--', label='Threshold')
        ax2.legend()
        ax2.set_title('Z-score Over Time')

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
