# Importing necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# Loading the digits dataset
data = load_digits()
X = data.data
y = data.target
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initializing the MLP (ANN) Classifier
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
# Making predictions on the test data
y_pred = mlp.predict(X_test)
# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"ANN Classifier Accuracy: {accuracy:.2f}")
# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Plotting the loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("ANN Training Loss Curve")
plt.show()
