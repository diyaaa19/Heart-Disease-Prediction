# Heart Disease Prediction

Machine learning project for predicting heart disease using multiple classification algorithms and ensemble learning techniques on the UCI Heart Disease dataset.
## Screenshots
<img width="1356" height="647" alt="image" src="https://github.com/user-attachments/assets/745a97c7-14ac-48fb-9701-7bb7a26010ce" />
<img width="1116" height="615" alt="image" src="https://github.com/user-attachments/assets/87fdd4bc-7c33-464e-803a-5e7e48d248bd" />

## Overview

The project compares traditional classification models with ensemble methods and evaluates their performance using test accuracy and 5-fold cross-validation.

The workflow includes:

* Data exploration and preprocessing
* Feature scaling using `StandardScaler`
* K-Nearest Neighbors
* Support Vector Classifier
* Decision Tree
* Random Forest
* Soft Voting
* Hard Voting
* Bagging
* Stacking
* Blending
* Random Forest hyperparameter tuning with `GridSearchCV`
* Model comparison and confusion matrix analysis
* Sample prediction using the best ensemble

## Results

The blending ensemble achieved the highest test accuracy of **96.15%**.

The final blending model combines:

* Random Forest
* Extra Trees
* Decision Tree
* SVM

and uses Logistic Regression as the meta-model.

The project also compares the proposed blending approach against baseline and ensemble models, along with results from selected prior studies.

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

## Project Structure

```text
heart-disease-prediction/
├── heart_disease_ensemble.ipynb
├── dataset.csv
├── blending_comparison.png
└── README.md
```

## Run the Project

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Open the notebook in Jupyter or Google Colab and run the cells sequentially.

The notebook expects the dataset to be available as:

```text
dataset.csv
```

## Example Prediction

The notebook includes a sample patient prediction using the final blending model, along with the predicted class and model confidence.

## Note

This project is intended for educational and machine learning experimentation purposes and should not be used as a medical diagnostic system.
