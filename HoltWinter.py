import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class CustomHoltWinters:
    def __init__(self, alpha, beta, gamma, seasonal_period):
        self.alpha = alpha  # Level smoothing parameter
        self.beta = beta    # Trend smoothing parameter
        self.gamma = gamma  # Seasonal smoothing parameter
        self.seasonal_period = seasonal_period
        self.level = None
        self.trend = None
        self.seasonals = None

    def initial_trend(self, data):
        trend = 0
        for i in range(self.seasonal_period):
            trend += (data[i + self.seasonal_period] - data[i]) / self.seasonal_period
        return trend / self.seasonal_period

    def initial_seasonal_components(self, data):
        seasonals = {}
        for i in range(self.seasonal_period):
            seasonals[i] = data[i] / (sum(data[:self.seasonal_period]) / self.seasonal_period)
        return seasonals

    def fit(self, data):
        self.level = sum(data[:self.seasonal_period]) / self.seasonal_period
        self.trend = self.initial_trend(data)
        self.seasonals = self.initial_seasonal_components(data)
        predictions = []
        for i in range(len(data)):
            m = i - self.seasonal_period
            if m >= 0:
                season = self.seasonals[m % self.seasonal_period]
                prediction = self.level + self.trend + season
            else:
                prediction = data[i]
            predictions.append(prediction)
            self.update(data[i], i)
        return predictions

    def update(self, data_point, t):
        last_level = self.level
        seasonal_index = t % self.seasonal_period
        self.level = self.alpha * (data_point / self.seasonals[seasonal_index]) + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        self.seasonals[seasonal_index] = self.gamma * (data_point / self.level) + (1 - self.gamma) * self.seasonals[seasonal_index]

    def predict(self, steps):
        forecasts = []
        for i in range(steps):
            season = self.seasonals[i % self.seasonal_period]
            forecast = self.level + self.trend * i + season
            forecasts.append(forecast)
        return forecasts

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
THRESHOLD = 5.0  # Threshold for detecting anomalies based on deviation from prediction
ALPHA = 0.5
BETA = 0.3
GAMMA = 0.1
SEASONAL_PERIOD = 50

# Initialize Holt-Winters model
holt_winters = CustomHoltWinters(alpha=ALPHA, beta=BETA, gamma=GAMMA, seasonal_period=SEASONAL_PERIOD)
holt_winters.fit(data[:SEASONAL_PERIOD * 2])  # Fit initial data

# Initialize variables
forecasted_values = []
deviations = []
anomalies = []
xs = []
ys = []
detected_anomaly_indices = []

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

def animate(i):
    if i < len(df):
        new_value = df['value'].iloc[i]
        xs.append(i)
        ys.append(new_value)

        # Holt-Winters prediction
        prediction = holt_winters.predict(1)[0]
        forecasted_values.append(prediction)
        deviation = abs(new_value - prediction)
        deviations.append(deviation)

        # Update the Holt-Winters model with the new value
        holt_winters.update(new_value, i)

        # Anomaly detection based on deviation
        is_anomaly = deviation > THRESHOLD
        anomalies.append(is_anomaly)

        if is_anomaly:
            detected_anomaly_indices.append(i)

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

        # Plot deviations
        ax2.plot(xs, deviations, label='Deviation')
        ax2.axhline(y=THRESHOLD, color='r', linestyle='--', label='Threshold')
        ax2.legend()
        ax2.set_title('Deviation from Prediction Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Deviation')

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
