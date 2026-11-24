{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO0zj4FAEvOGJDpFw4kRbxZ",
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
        "<a href=\"https://colab.research.google.com/github/madhuvasucv-hash/Madhumitha-Vasudevan/blob/main/1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# FIND-S Algorithm\n",
        "\n",
        "import csv\n",
        "\n",
        "# Training dataset\n",
        "data = [\n",
        "    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],\n",
        "    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],\n",
        "    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],\n",
        "    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']\n",
        "]\n",
        "\n",
        "# Extract attributes and target\n",
        "num_attributes = len(data[0]) - 1\n",
        "\n",
        "# Initialize hypothesis\n",
        "hypothesis = ['0'] * num_attributes\n",
        "\n",
        "print(\"Training Examples:\")\n",
        "for row in data:\n",
        "    print(row)\n",
        "\n",
        "# Apply FIND-S algorithm\n",
        "for example in data:\n",
        "    if example[-1] == 'Yes':\n",
        "        for i in range(num_attributes):\n",
        "            if hypothesis[i] == '0':\n",
        "                hypothesis[i] = example[i]\n",
        "            elif hypothesis[i] != example[i]:\n",
        "                hypothesis[i] = '?'\n",
        "\n",
        "        print(\"Current Hypothesis:\", hypothesis)\n",
        "\n",
        "# Display final hypothesis\n",
        "print(\"\\nFinal Most Specific Hypothesis:\")\n",
        "print(hypothesis)\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "qOrEqDNvyGl0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "68c0c570-2ed2-4f63-fb20-7f75ef883a42"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Examples:\n",
            "['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']\n",
            "['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']\n",
            "['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']\n",
            "['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']\n",
            "Current Hypothesis: ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']\n",
            "Current Hypothesis: ['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']\n",
            "Current Hypothesis: ['Sunny', 'Warm', '?', 'Strong', '?', '?']\n",
            "\n",
            "Final Most Specific Hypothesis:\n",
            "['Sunny', 'Warm', '?', 'Strong', '?', '?']\n"
          ]
        }
      ]
    }
  ]
}