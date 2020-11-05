# Churn-modelling{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of artificial_neural_network.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lP6JLo1tGNBg"
      },
      "source": [
        "# Artificial Neural Network"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gWZyYmS_UE_L"
      },
      "source": [
        "### Importing the libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MxkJoQBkUIHC"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import tensorflow as tf"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1E0Q3aoKUCRX"
      },
      "source": [
        "## Part 1 - Data Preprocessing"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cKWAkFVGUU0Z"
      },
      "source": [
        "### Importing the dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MXUkhkMfU4wq"
      },
      "source": [
        "dataset = pd.read_csv('Churn_Modelling.csv')\n",
        "X = dataset.iloc[:, 3:-1].values\n",
        "y = dataset.iloc[:, -1].values"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "N6bQ0UgSU-NJ"
      },
      "source": [
        "### Encoding categorical data"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "le5MJreAbW52"
      },
      "source": [
        "Label Encoding the \"Gender\" column"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PxVKWXxLbczC"
      },
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "le = LabelEncoder()\n",
        "X[:, 2] = le.fit_transform(X[:, 2])"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CUxGZezpbMcb"
      },
      "source": [
        "One Hot Encoding the \"Geography\" column"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AMXC8-KMVirw"
      },
      "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')\n",
        "X = np.array(ct.fit_transform(X))"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vHol938cW8zd"
      },
      "source": [
        "### Splitting the dataset into the Training set and Test set"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Z-TDt0Y_XEfc"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RE_FcHyfV3TQ"
      },
      "source": [
        "### Feature Scaling"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ViCrE00rV8Sk"
      },
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "sc = StandardScaler()\n",
        "X_train = sc.fit_transform(X_train)\n",
        "X_test = sc.transform(X_test)"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-zfEzkRVXIwF"
      },
      "source": [
        "## Part 2 - Building the ANN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KvdeScabXtlB"
      },
      "source": [
        "### Initializing the ANN"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3dtrScHxXQox"
      },
      "source": [
        "ann = tf.keras.models.Sequential()"
      ],
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rP6urV6SX7kS"
      },
      "source": [
        "### Adding the input layer and the first hidden layer"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bppGycBXYCQr"
      },
      "source": [
        "ann.add(tf.keras.layers.Dense(units=6, activation='relu'))"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BELWAc_8YJze"
      },
      "source": [
        "### Adding the second hidden layer"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JneR0u0sYRTd"
      },
      "source": [
        "ann.add(tf.keras.layers.Dense(units=6, activation='relu'))"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OyNEe6RXYcU4"
      },
      "source": [
        "### Adding the output layer"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Cn3x41RBYfvY"
      },
      "source": [
        "ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JT4u2S1_Y4WG"
      },
      "source": [
        "## Part 3 - Training the ANN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8GWlJChhY_ZI"
      },
      "source": [
        "### Compiling the ANN"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fG3RrwDXZEaS"
      },
      "source": [
        "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0QR_G5u7ZLSM"
      },
      "source": [
        "### Training the ANN on the Training set"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nHZ-LKv_ZRb3",
        "outputId": "bb4d8860-2e10-4d97-cce8-196f7a32cca8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        }
      },
      "source": [
        "ann.fit(X_train, y_train, batch_size = 32, epochs = 100)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.5920 - accuracy: 0.7255\n",
            "Epoch 2/100\n",
            "250/250 [==============================] - 0s 913us/step - loss: 0.4643 - accuracy: 0.7971\n",
            "Epoch 3/100\n",
            "250/250 [==============================] - 0s 936us/step - loss: 0.4399 - accuracy: 0.8020\n",
            "Epoch 4/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.4291 - accuracy: 0.8083\n",
            "Epoch 5/100\n",
            "250/250 [==============================] - 0s 932us/step - loss: 0.4186 - accuracy: 0.8202\n",
            "Epoch 6/100\n",
            "250/250 [==============================] - 0s 965us/step - loss: 0.4115 - accuracy: 0.8245\n",
            "Epoch 7/100\n",
            "250/250 [==============================] - 0s 936us/step - loss: 0.4066 - accuracy: 0.8284\n",
            "Epoch 8/100\n",
            "250/250 [==============================] - 0s 947us/step - loss: 0.4027 - accuracy: 0.8291\n",
            "Epoch 9/100\n",
            "250/250 [==============================] - 0s 952us/step - loss: 0.3994 - accuracy: 0.8294\n",
            "Epoch 10/100\n",
            "250/250 [==============================] - 0s 905us/step - loss: 0.3967 - accuracy: 0.8313\n",
            "Epoch 11/100\n",
            "250/250 [==============================] - 0s 929us/step - loss: 0.3935 - accuracy: 0.8307\n",
            "Epoch 12/100\n",
            "250/250 [==============================] - 0s 934us/step - loss: 0.3907 - accuracy: 0.8300\n",
            "Epoch 13/100\n",
            "250/250 [==============================] - 0s 957us/step - loss: 0.3878 - accuracy: 0.8320\n",
            "Epoch 14/100\n",
            "250/250 [==============================] - 0s 926us/step - loss: 0.3845 - accuracy: 0.8324\n",
            "Epoch 15/100\n",
            "250/250 [==============================] - 0s 876us/step - loss: 0.3814 - accuracy: 0.8324\n",
            "Epoch 16/100\n",
            "250/250 [==============================] - 0s 923us/step - loss: 0.3779 - accuracy: 0.8357\n",
            "Epoch 17/100\n",
            "250/250 [==============================] - 0s 966us/step - loss: 0.3743 - accuracy: 0.8425\n",
            "Epoch 18/100\n",
            "250/250 [==============================] - 0s 924us/step - loss: 0.3704 - accuracy: 0.8453\n",
            "Epoch 19/100\n",
            "250/250 [==============================] - 0s 923us/step - loss: 0.3662 - accuracy: 0.8487\n",
            "Epoch 20/100\n",
            "250/250 [==============================] - 0s 925us/step - loss: 0.3625 - accuracy: 0.8520\n",
            "Epoch 21/100\n",
            "250/250 [==============================] - 0s 947us/step - loss: 0.3581 - accuracy: 0.8545\n",
            "Epoch 22/100\n",
            "250/250 [==============================] - 0s 977us/step - loss: 0.3537 - accuracy: 0.8561\n",
            "Epoch 23/100\n",
            "250/250 [==============================] - 0s 953us/step - loss: 0.3502 - accuracy: 0.8570\n",
            "Epoch 24/100\n",
            "250/250 [==============================] - 0s 983us/step - loss: 0.3479 - accuracy: 0.8575\n",
            "Epoch 25/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3456 - accuracy: 0.8597\n",
            "Epoch 26/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3445 - accuracy: 0.8577\n",
            "Epoch 27/100\n",
            "250/250 [==============================] - 0s 894us/step - loss: 0.3433 - accuracy: 0.8583\n",
            "Epoch 28/100\n",
            "250/250 [==============================] - 0s 906us/step - loss: 0.3422 - accuracy: 0.8605\n",
            "Epoch 29/100\n",
            "250/250 [==============================] - 0s 921us/step - loss: 0.3419 - accuracy: 0.8609\n",
            "Epoch 30/100\n",
            "250/250 [==============================] - 0s 980us/step - loss: 0.3413 - accuracy: 0.8608\n",
            "Epoch 31/100\n",
            "250/250 [==============================] - 0s 966us/step - loss: 0.3411 - accuracy: 0.8608\n",
            "Epoch 32/100\n",
            "250/250 [==============================] - 0s 965us/step - loss: 0.3404 - accuracy: 0.8609\n",
            "Epoch 33/100\n",
            "250/250 [==============================] - 0s 877us/step - loss: 0.3398 - accuracy: 0.8614\n",
            "Epoch 34/100\n",
            "250/250 [==============================] - 0s 974us/step - loss: 0.3396 - accuracy: 0.8615\n",
            "Epoch 35/100\n",
            "250/250 [==============================] - 0s 985us/step - loss: 0.3393 - accuracy: 0.8616\n",
            "Epoch 36/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3389 - accuracy: 0.8621\n",
            "Epoch 37/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3391 - accuracy: 0.8616\n",
            "Epoch 38/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3385 - accuracy: 0.8622\n",
            "Epoch 39/100\n",
            "250/250 [==============================] - 0s 932us/step - loss: 0.3383 - accuracy: 0.8626\n",
            "Epoch 40/100\n",
            "250/250 [==============================] - 0s 925us/step - loss: 0.3380 - accuracy: 0.8637\n",
            "Epoch 41/100\n",
            "250/250 [==============================] - 0s 955us/step - loss: 0.3374 - accuracy: 0.8625\n",
            "Epoch 42/100\n",
            "250/250 [==============================] - 0s 893us/step - loss: 0.3375 - accuracy: 0.8636\n",
            "Epoch 43/100\n",
            "250/250 [==============================] - 0s 927us/step - loss: 0.3370 - accuracy: 0.8633\n",
            "Epoch 44/100\n",
            "250/250 [==============================] - 0s 920us/step - loss: 0.3373 - accuracy: 0.8633\n",
            "Epoch 45/100\n",
            "250/250 [==============================] - 0s 906us/step - loss: 0.3369 - accuracy: 0.8625\n",
            "Epoch 46/100\n",
            "250/250 [==============================] - 0s 917us/step - loss: 0.3371 - accuracy: 0.8619\n",
            "Epoch 47/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3368 - accuracy: 0.8625\n",
            "Epoch 48/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3366 - accuracy: 0.8629\n",
            "Epoch 49/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3362 - accuracy: 0.8620\n",
            "Epoch 50/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3363 - accuracy: 0.8633\n",
            "Epoch 51/100\n",
            "250/250 [==============================] - 0s 937us/step - loss: 0.3363 - accuracy: 0.8622\n",
            "Epoch 52/100\n",
            "250/250 [==============================] - 0s 966us/step - loss: 0.3355 - accuracy: 0.8626\n",
            "Epoch 53/100\n",
            "250/250 [==============================] - 0s 950us/step - loss: 0.3356 - accuracy: 0.8630\n",
            "Epoch 54/100\n",
            "250/250 [==============================] - 0s 924us/step - loss: 0.3355 - accuracy: 0.8633\n",
            "Epoch 55/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3357 - accuracy: 0.8631\n",
            "Epoch 56/100\n",
            "250/250 [==============================] - 0s 976us/step - loss: 0.3355 - accuracy: 0.8634\n",
            "Epoch 57/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3351 - accuracy: 0.8633\n",
            "Epoch 58/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3351 - accuracy: 0.8626\n",
            "Epoch 59/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3349 - accuracy: 0.8648\n",
            "Epoch 60/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3352 - accuracy: 0.8627\n",
            "Epoch 61/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3347 - accuracy: 0.8629\n",
            "Epoch 62/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3347 - accuracy: 0.8659\n",
            "Epoch 63/100\n",
            "250/250 [==============================] - 0s 954us/step - loss: 0.3346 - accuracy: 0.8636\n",
            "Epoch 64/100\n",
            "250/250 [==============================] - 0s 924us/step - loss: 0.3348 - accuracy: 0.8648\n",
            "Epoch 65/100\n",
            "250/250 [==============================] - 0s 881us/step - loss: 0.3345 - accuracy: 0.8636\n",
            "Epoch 66/100\n",
            "250/250 [==============================] - 0s 899us/step - loss: 0.3344 - accuracy: 0.8621\n",
            "Epoch 67/100\n",
            "250/250 [==============================] - 0s 961us/step - loss: 0.3341 - accuracy: 0.8637\n",
            "Epoch 68/100\n",
            "250/250 [==============================] - 0s 895us/step - loss: 0.3340 - accuracy: 0.8633\n",
            "Epoch 69/100\n",
            "250/250 [==============================] - 0s 991us/step - loss: 0.3337 - accuracy: 0.8652\n",
            "Epoch 70/100\n",
            "250/250 [==============================] - 0s 911us/step - loss: 0.3347 - accuracy: 0.8639\n",
            "Epoch 71/100\n",
            "250/250 [==============================] - 0s 962us/step - loss: 0.3344 - accuracy: 0.8637\n",
            "Epoch 72/100\n",
            "250/250 [==============================] - 0s 985us/step - loss: 0.3337 - accuracy: 0.8639\n",
            "Epoch 73/100\n",
            "250/250 [==============================] - 0s 974us/step - loss: 0.3340 - accuracy: 0.8648\n",
            "Epoch 74/100\n",
            "250/250 [==============================] - 0s 885us/step - loss: 0.3336 - accuracy: 0.8654\n",
            "Epoch 75/100\n",
            "250/250 [==============================] - 0s 922us/step - loss: 0.3339 - accuracy: 0.8635\n",
            "Epoch 76/100\n",
            "250/250 [==============================] - 0s 885us/step - loss: 0.3335 - accuracy: 0.8629\n",
            "Epoch 77/100\n",
            "250/250 [==============================] - 0s 971us/step - loss: 0.3338 - accuracy: 0.8634\n",
            "Epoch 78/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3337 - accuracy: 0.8654\n",
            "Epoch 79/100\n",
            "250/250 [==============================] - 0s 905us/step - loss: 0.3340 - accuracy: 0.8654\n",
            "Epoch 80/100\n",
            "250/250 [==============================] - 0s 943us/step - loss: 0.3338 - accuracy: 0.8656\n",
            "Epoch 81/100\n",
            "250/250 [==============================] - 0s 941us/step - loss: 0.3336 - accuracy: 0.8635\n",
            "Epoch 82/100\n",
            "250/250 [==============================] - 0s 899us/step - loss: 0.3342 - accuracy: 0.8645\n",
            "Epoch 83/100\n",
            "250/250 [==============================] - 0s 907us/step - loss: 0.3334 - accuracy: 0.8645\n",
            "Epoch 84/100\n",
            "250/250 [==============================] - 0s 893us/step - loss: 0.3334 - accuracy: 0.8639\n",
            "Epoch 85/100\n",
            "250/250 [==============================] - 0s 936us/step - loss: 0.3334 - accuracy: 0.8646\n",
            "Epoch 86/100\n",
            "250/250 [==============================] - 0s 924us/step - loss: 0.3332 - accuracy: 0.8644\n",
            "Epoch 87/100\n",
            "250/250 [==============================] - 0s 931us/step - loss: 0.3334 - accuracy: 0.8639\n",
            "Epoch 88/100\n",
            "250/250 [==============================] - 0s 885us/step - loss: 0.3339 - accuracy: 0.8646\n",
            "Epoch 89/100\n",
            "250/250 [==============================] - 0s 972us/step - loss: 0.3336 - accuracy: 0.8633\n",
            "Epoch 90/100\n",
            "250/250 [==============================] - 0s 946us/step - loss: 0.3330 - accuracy: 0.8639\n",
            "Epoch 91/100\n",
            "250/250 [==============================] - 0s 920us/step - loss: 0.3329 - accuracy: 0.8649\n",
            "Epoch 92/100\n",
            "250/250 [==============================] - 0s 969us/step - loss: 0.3336 - accuracy: 0.8654\n",
            "Epoch 93/100\n",
            "250/250 [==============================] - 0s 956us/step - loss: 0.3337 - accuracy: 0.8645\n",
            "Epoch 94/100\n",
            "250/250 [==============================] - 0s 1ms/step - loss: 0.3330 - accuracy: 0.8646\n",
            "Epoch 95/100\n",
            "250/250 [==============================] - 0s 968us/step - loss: 0.3332 - accuracy: 0.8627\n",
            "Epoch 96/100\n",
            "250/250 [==============================] - 0s 902us/step - loss: 0.3329 - accuracy: 0.8626\n",
            "Epoch 97/100\n",
            "250/250 [==============================] - 0s 915us/step - loss: 0.3331 - accuracy: 0.8633\n",
            "Epoch 98/100\n",
            "250/250 [==============================] - 0s 903us/step - loss: 0.3330 - accuracy: 0.8649\n",
            "Epoch 99/100\n",
            "250/250 [==============================] - 0s 882us/step - loss: 0.3328 - accuracy: 0.8630\n",
            "Epoch 100/100\n",
            "250/250 [==============================] - 0s 862us/step - loss: 0.3327 - accuracy: 0.8639\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tensorflow.python.keras.callbacks.History at 0x7f3287294630>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tJj5k2MxZga3"
      },
      "source": [
        "## Part 4 - Making the predictions and evaluating the model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "84QFoqGYeXHL"
      },
      "source": [
        "### Predicting the result of a single observation"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CGRo3eacgDdC"
      },
      "source": [
        "\n",
        "\n",
        "\n",
        "Geography: France\n",
        "\n",
        "Credit Score: 600\n",
        "\n",
        "Gender: Male\n",
        "\n",
        "Age: 40 years old\n",
        "\n",
        "Tenure: 3 years\n",
        "\n",
        "Balance: \\$ 60000\n",
        "\n",
        "Number of Products: 2\n",
        "\n",
        "Does this customer have a credit card? Yes\n",
        "\n",
        "Is this customer an Active Member: Yes\n",
        "\n",
        "Estimated Salary: \\$ 50000\n",
        "\n",
        "So, should we say goodbye to that customer?"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZhU1LTgPg-kH"
      },
      "source": [
        "**Solution**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2d8IoCCkeWGL",
        "outputId": "d58dd9cd-7ab1-4052-80ff-d289e21c5a52",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[False]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u7yx47jPZt11"
      },
      "source": [
        "### Predicting the Test set results"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nIyEeQdRZwgs",
        "outputId": "2053dfd4-8ec3-4dc8-94cc-f032b57e5cbe",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 134
        }
      },
      "source": [
        "y_pred = ann.predict(X_test)\n",
        "y_pred = (y_pred > 0.5)\n",
        "print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[0 0]\n",
            " [0 1]\n",
            " [0 0]\n",
            " ...\n",
            " [0 0]\n",
            " [0 0]\n",
            " [0 0]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "o0oyfLWoaEGw"
      },
      "source": [
        "### Making the Confusion Matrix"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ci6K_r6LaF6P",
        "outputId": "9fcd5216-7d49-48f5-b89b-9d9a3ce5bc1a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 67
        }
      },
      "source": [
        "from sklearn.metrics import confusion_matrix, accuracy_score\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "print(cm)\n",
        "accuracy_score(y_test, y_pred)"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[1519   76]\n",
            " [ 204  201]]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.86"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        }
      ]
    }
  ]
}
