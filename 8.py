{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMHO7U3upFaAhSZgaOIfZ39",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/madhuvasucv-hash/Madhumitha-Vasudevan/blob/main/8.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from sklearn.datasets import fetch_california_housing\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
        "\n",
        "class LinearRegressionScratch:\n",
        "    def __init__(self, learning_rate=0.01, epochs=1000):\n",
        "        \"\"\"\n",
        "        Initializes the Linear Regression model.\n",
        "        :param learning_rate: Step size for gradient descent\n",
        "        :param epochs: Number of training iterations\n",
        "        \"\"\"\n",
        "        self.lr = learning_rate\n",
        "        self.epochs = epochs\n",
        "        self.weights = None\n",
        "        self.bias = None\n",
        "\n",
        "    @staticmethod\n",
        "    def _compute_mse(y_true, y_pred):\n",
        "        \"\"\"Computes Mean Squared Error (MSE).\"\"\"\n",
        "        return np.mean((y_true - y_pred) ** 2)\n",
        "\n",
        "    def fit(self, X, y, print_every=200):\n",
        "        \"\"\"Trains the model using Batch Gradient Descent.\"\"\"\n",
        "        num_samples, num_features = X.shape\n",
        "        self.weights = np.zeros(num_features)\n",
        "        self.bias = 0.0\n",
        "\n",
        "        for epoch in range(1, self.epochs + 1):\n",
        "            # 1. Forward prediction: y_hat = Xw + b\n",
        "            y_pred = np.dot(X, self.weights) + self.bias\n",
        "\n",
        "            # 2. Compute gradients of MSE with respect to weights and bias\n",
        "            dw = (2 / num_samples) * np.dot(X.T, (y_pred - y))\n",
        "            db = (2 / num_samples) * np.sum(y_pred - y)\n",
        "\n",
        "            # 3. Update parameters\n",
        "            self.weights -= self.lr * dw\n",
        "            self.bias -= self.lr * db\n",
        "\n",
        "            # Print training progress\n",
        "            if print_every and epoch % print_every == 0:\n",
        "                mse = self._compute_mse(y, y_pred)\n",
        "                print(f\"Epoch {epoch:4d} | MSE Loss: {mse:.4f}\")\n",
        "\n",
        "    def predict(self, X):\n",
        "        \"\"\"Predicts continuous target values for input matrix X.\"\"\"\n",
        "        return np.dot(X, self.weights) + self.bias\n",
        "\n",
        "\n",
        "# =====================================================================\n",
        "# Demonstration on the California Housing Dataset\n",
        "# =====================================================================\n",
        "if __name__ == \"__main__\":\n",
        "    # 1. Load dataset (Target: Median house value in $100,000s)\n",
        "    housing = fetch_california_housing()\n",
        "    X, y = housing.data, housing.target\n",
        "\n",
        "    # 2. Split into training (80%) and testing (20%) sets\n",
        "    X_train, X_test, y_train, y_test = train_test_split(\n",
        "        X, y, test_size=0.2, random_state=42\n",
        "    )\n",
        "\n",
        "    # 3. Standardize features (prevents exploding gradients across different feature scales)\n",
        "    scaler = StandardScaler()\n",
        "    X_train_scaled = scaler.fit_transform(X_train)\n",
        "    X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "    # 4. Initialize and train the Linear Regression model\n",
        "    print(\"=== TRAINING LINEAR REGRESSION ===\\n\")\n",
        "    model = LinearRegressionScratch(learning_rate=0.05, epochs=1000)\n",
        "    model.fit(X_train_scaled, y_train, print_every=200)\n",
        "\n",
        "    # 5. Evaluate on unseen test data\n",
        "    y_pred = model.predict(X_test_scaled)\n",
        "\n",
        "    mse = mean_squared_error(y_test, y_pred)\n",
        "    rmse = np.sqrt(mse)\n",
        "    mae = mean_absolute_error(y_test, y_pred)\n",
        "    r2 = r2_score(y_test, y_pred)\n",
        "\n",
        "    print(\"\\n=== EVALUATION RESULTS ===\")\n",
        "    print(f\"Mean Squared Error (MSE)      : {mse:.4f}\")\n",
        "    print(f\"Root Mean Squared Error (RMSE): {rmse:.4f}\")\n",
        "    print(f\"Mean Absolute Error (MAE)     : {mae:.4f}\")\n",
        "    print(f\"R² Score (Variance Explained) : {r2:.4f}\\n\")\n",
        "\n",
        "    # 6. Show sample predictions vs actual values\n",
        "    print(\"=== SAMPLE PREDICTIONS (in $100,000s) ===\")\n",
        "    for i in range(5):\n",
        "        print(f\"Sample {i+1} | Actual: {y_test[i]:.3f} | Predicted: {y_pred[i]:.3f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9aYLXDgNtwPe",
        "outputId": "de3832a5-dbce-453f-dcf0-146e56654230"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "=== TRAINING LINEAR REGRESSION ===\n",
            "\n",
            "Epoch  200 | MSE Loss: 0.5246\n",
            "Epoch  400 | MSE Loss: 0.5184\n",
            "Epoch  600 | MSE Loss: 0.5180\n",
            "Epoch  800 | MSE Loss: 0.5179\n",
            "Epoch 1000 | MSE Loss: 0.5179\n",
            "\n",
            "=== EVALUATION RESULTS ===\n",
            "Mean Squared Error (MSE)      : 0.5560\n",
            "Root Mean Squared Error (RMSE): 0.7457\n",
            "Mean Absolute Error (MAE)     : 0.5332\n",
            "R² Score (Variance Explained) : 0.5757\n",
            "\n",
            "=== SAMPLE PREDICTIONS (in $100,000s) ===\n",
            "Sample 1 | Actual: 0.477 | Predicted: 0.720\n",
            "Sample 2 | Actual: 0.458 | Predicted: 1.763\n",
            "Sample 3 | Actual: 5.000 | Predicted: 2.710\n",
            "Sample 4 | Actual: 2.186 | Predicted: 2.838\n",
            "Sample 5 | Actual: 2.780 | Predicted: 2.603\n"
          ]
        }
      ]
    }
  ]
}