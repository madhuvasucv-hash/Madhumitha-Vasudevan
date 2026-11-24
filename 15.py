{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPYVtPl+AJM5Bk41hQpNvu9",
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
        "<a href=\"https://colab.research.google.com/github/madhuvasucv-hash/Madhumitha-Vasudevan/blob/main/15.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "\n",
        "# 1. Load the Iris Dataset\n",
        "iris = load_iris()\n",
        "X = iris.data\n",
        "y = iris.target\n",
        "feature_names = iris.feature_names\n",
        "target_names = iris.target_names\n",
        "\n",
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
        "# 3. Initialize and Train the Gaussian Naive Bayes Classifier\n",
        "# Note: Feature scaling is not strictly required for GaussianNB because\n",
        "# each feature's mean and variance are computed independently per class.\n",
        "nb_classifier = GaussianNB()\n",
        "nb_classifier.fit(X_train, y_train)\n",
        "\n",
        "# 4. Make Predictions and Evaluate Performance\n",
        "y_pred = nb_classifier.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred) * 100\n",
        "\n",
        "print(\"=== MODEL EVALUATION (GAUSSIAN NAIVE BAYES) ===\")\n",
        "print(f\"Test Accuracy: {accuracy:.2f}%\\n\")\n",
        "\n",
        "print(\"Confusion Matrix:\")\n",
        "print(confusion_matrix(y_test, y_pred), \"\\n\")\n",
        "\n",
        "print(\"Classification Report:\")\n",
        "print(classification_report(y_test, y_pred, target_names=target_names))\n",
        "\n",
        "# 5. Inspect Learned Gaussian Parameters (Class Means)\n",
        "means_df = pd.DataFrame(\n",
        "    nb_classifier.theta_,\n",
        "    index=target_names,\n",
        "    columns=feature_names\n",
        ")\n",
        "print(\"=== LEARNED FEATURE MEANS PER SPECIES ===\")\n",
        "print(means_df.round(3), \"\\n\")\n",
        "\n",
        "# 6. Predict Species for a New Unseen Flower Sample\n",
        "# Measurements: [sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)]\n",
        "new_flower = np.array([[6.0, 2.9, 4.5, 1.5]])\n",
        "predicted_class_idx = nb_classifier.predict(new_flower)[0]\n",
        "predicted_species = target_names[predicted_class_idx]\n",
        "probabilities = nb_classifier.predict_proba(new_flower)[0]\n",
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
        "id": "x2OBk7I5yDjl",
        "outputId": "6a6d3071-562a-4f3a-b531-40eb38060c98"
      },
      "execution_count": 19,
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
            "=== MODEL EVALUATION (GAUSSIAN NAIVE BAYES) ===\n",
            "Test Accuracy: 96.67%\n",
            "\n",
            "Confusion Matrix:\n",
            "[[10  0  0]\n",
            " [ 0  9  1]\n",
            " [ 0  0 10]] \n",
            "\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "      setosa       1.00      1.00      1.00        10\n",
            "  versicolor       1.00      0.90      0.95        10\n",
            "   virginica       0.91      1.00      0.95        10\n",
            "\n",
            "    accuracy                           0.97        30\n",
            "   macro avg       0.97      0.97      0.97        30\n",
            "weighted avg       0.97      0.97      0.97        30\n",
            "\n",
            "=== LEARNED FEATURE MEANS PER SPECIES ===\n",
            "            sepal length (cm)  sepal width (cm)  petal length (cm)  \\\n",
            "setosa                  4.985             3.415              1.477   \n",
            "versicolor              5.930             2.750              4.252   \n",
            "virginica               6.610             2.980              5.580   \n",
            "\n",
            "            petal width (cm)  \n",
            "setosa                 0.255  \n",
            "versicolor             1.320  \n",
            "virginica              2.040   \n",
            "\n",
            "=== NEW SAMPLE PREDICTION ===\n",
            "Input Measurements : [6.  2.9 4.5 1.5]\n",
            "Predicted Species  : versicolor\n",
            "  Probability (setosa    ): 0.00%\n",
            "  Probability (versicolor): 99.06%\n",
            "  Probability (virginica ): 0.94%\n"
          ]
        }
      ]
    }
  ]
}