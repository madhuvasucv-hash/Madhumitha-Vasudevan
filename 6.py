{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPBow1ZysjzxHxpPfTNfEjq",
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
        "<a href=\"https://colab.research.google.com/github/madhuvasucv-hash/Madhumitha-Vasudevan/blob/main/6.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.metrics import confusion_matrix, accuracy_score, classification_report\n",
        "\n",
        "# Dataset\n",
        "X = np.array([\n",
        "    [25, 40000],\n",
        "    [30, 50000],\n",
        "    [35, 60000],\n",
        "    [40, 80000],\n",
        "    [45, 90000],\n",
        "    [20, 30000],\n",
        "    [50, 100000],\n",
        "    [23, 35000],\n",
        "    [33, 55000],\n",
        "    [48, 95000]\n",
        "])\n",
        "\n",
        "# Target values\n",
        "y = np.array([\n",
        "    'No', 'No', 'Yes', 'Yes', 'Yes',\n",
        "    'No', 'Yes', 'No', 'Yes', 'Yes'\n",
        "])\n",
        "\n",
        "# Split dataset into training and testing\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.3, random_state=42\n",
        ")\n",
        "\n",
        "# Create Naive Bayes classifier\n",
        "model = GaussianNB()\n",
        "\n",
        "# Train the model\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predict the output\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Display results\n",
        "print(\"Actual Values:\", y_test)\n",
        "print(\"Predicted Values:\", y_pred)\n",
        "\n",
        "# Confusion Matrix\n",
        "cm = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])\n",
        "print(\"\\nConfusion Matrix:\")\n",
        "print(cm)\n",
        "\n",
        "# Calculate Accuracy\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(\"\\nAccuracy:\", accuracy * 100, \"%\")\n",
        "\n",
        "# Classification Report\n",
        "print(\"\\nClassification Report:\")\n",
        "print(classification_report(y_test, y_pred, zero_division=0))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T1DcpNAzsINU",
        "outputId": "413550e9-dba9-4b0b-8562-7067df0778f1"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Actual Values: ['Yes' 'No' 'No']\n",
            "Predicted Values: ['Yes' 'Yes' 'No']\n",
            "\n",
            "Confusion Matrix:\n",
            "[[1 1]\n",
            " [0 1]]\n",
            "\n",
            "Accuracy: 66.66666666666666 %\n",
            "\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "          No       1.00      0.50      0.67         2\n",
            "         Yes       0.50      1.00      0.67         1\n",
            "\n",
            "    accuracy                           0.67         3\n",
            "   macro avg       0.75      0.75      0.67         3\n",
            "weighted avg       0.83      0.67      0.67         3\n",
            "\n"
          ]
        }
      ]
    }
  ]
}