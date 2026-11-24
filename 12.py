{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM+Q5vJ9OBWrd8zPf5Nyfjg",
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
        "<a href=\"https://colab.research.google.com/github/madhuvasucv-hash/Madhumitha-Vasudevan/blob/main/12.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "\n",
        "# 1. Load the Iris Dataset\n",
        "iris = load_iris()\n",
        "X = iris.data\n",
        "y = iris.target\n",
        "feature_names = iris.feature_names\n",
        "target_names = iris.target_names\n",
        "\n",
        "# Display dataset summary\n",
        "df = pd.DataFrame(X, columns=feature_names)\n",
        "df[\"species\"] = [target_names[i] for i in y]\n",
        "print(\"=== DATASET SAMPLE (FIRST 5 ROWS) ===\")\n",
        "print(df.head(), \"\\n\")\n",
        "\n",
        "# 2. Split into Training (80%) and Testing (20%) Sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "# 3. Feature Scaling (Standardization)\n",
        "# Crucial for KNN because distance calculations are sensitive to feature scales\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "# 4. Initialize and Train the KNN Classifier\n",
        "k = 5\n",
        "knn = KNeighborsClassifier(n_neighbors=k, metric=\"euclidean\")\n",
        "knn.fit(X_train_scaled, y_train)\n",
        "\n",
        "# 5. Make Predictions and Evaluate Model Performance\n",
        "y_pred = knn.predict(X_test_scaled)\n",
        "accuracy = accuracy_score(y_test, y_pred) * 100\n",
        "\n",
        "print(f\"=== MODEL EVALUATION (k={k}) ===\")\n",
        "print(f\"Test Accuracy: {accuracy:.2f}%\\n\")\n",
        "\n",
        "print(\"Confusion Matrix:\")\n",
        "print(confusion_matrix(y_test, y_pred), \"\\n\")\n",
        "\n",
        "print(\"Classification Report:\")\n",
        "print(classification_report(y_test, y_pred, target_names=target_names))\n",
        "\n",
        "# 6. Predict Species for a New Unseen Flower Sample\n",
        "# Measurements: [sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)]\n",
        "new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])\n",
        "new_flower_scaled = scaler.transform(new_flower)\n",
        "\n",
        "predicted_class_idx = knn.predict(new_flower_scaled)[0]\n",
        "predicted_species = target_names[predicted_class_idx]\n",
        "probabilities = knn.predict_proba(new_flower_scaled)[0]\n",
        "\n",
        "print(\"=== NEW SAMPLE PREDICTION ===\")\n",
        "print(f\"Input Measurements : {new_flower[0]}\")\n",
        "print(f\"Predicted Species  : {predicted_species}\")\n",
        "for species, prob in zip(target_names, probabilities):\n",
        "    print(f\"  Probability ({species:10s}): {prob * 100:.2f}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "59W52OYAwmUd",
        "outputId": "468b0cac-760c-4463-aebc-813c9ab53c48"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "=== DATASET SAMPLE (FIRST 5 ROWS) ===\n",
            "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
            "0                5.1               3.5                1.4               0.2   \n",
            "1                4.9               3.0                1.4               0.2   \n",
            "2                4.7               3.2                1.3               0.2   \n",
            "3                4.6               3.1                1.5               0.2   \n",
            "4                5.0               3.6                1.4               0.2   \n",
            "\n",
            "  species  \n",
            "0  setosa  \n",
            "1  setosa  \n",
            "2  setosa  \n",
            "3  setosa  \n",
            "4  setosa   \n",
            "\n",
            "=== MODEL EVALUATION (k=5) ===\n",
            "Test Accuracy: 93.33%\n",
            "\n",
            "Confusion Matrix:\n",
            "[[10  0  0]\n",
            " [ 0 10  0]\n",
            " [ 0  2  8]] \n",
            "\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "      setosa       1.00      1.00      1.00        10\n",
            "  versicolor       0.83      1.00      0.91        10\n",
            "   virginica       1.00      0.80      0.89        10\n",
            "\n",
            "    accuracy                           0.93        30\n",
            "   macro avg       0.94      0.93      0.93        30\n",
            "weighted avg       0.94      0.93      0.93        30\n",
            "\n",
            "=== NEW SAMPLE PREDICTION ===\n",
            "Input Measurements : [5.1 3.5 1.4 0.2]\n",
            "Predicted Species  : setosa\n",
            "  Probability (setosa    ): 100.00%\n",
            "  Probability (versicolor): 0.00%\n",
            "  Probability (virginica ): 0.00%\n"
          ]
        }
      ]
    }
  ]
}