# Project-Folder
ML Assignment-2
A. Problem Statement:
The objective of this project is to build and compare multiple machine learning classification models on a public classification dataset, evaluate their performance using standard classification metrics, and deploy the solution as an interactive Streamlit web application.

B. Dataset Description:
Dataset used: Wisconsin Diagnostic Breast Cancer Dataset from UCI, available through sklearn.datasets.load_breast_cancer().

- Problem type: Binary classification
- Number of instances: 569
- Number of input features: 30
- Target classes: malignant and benign
- Target column used in this project: target
- Feature examples: mean radius, mean texture, mean perimeter, mean area, mean smoothness, worst radius, worst texture, worst perimeter, worst area

C. GitHub Repository Link:
https://github.com/2025ac05891-bot/Project-Folder/

Live Streamlit App Link:
(https://glorious-pancake-96prg946657295wv.github.dev/)

D. Models Used and Evaluation Metrics

The assignment document lists five required models but also mentions "all 6 models". Therefore, this solution implements the five explicitly listed models and adds Support Vector Machine as a sixth model.

ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC
Logistic Regression | 0.986 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.97
Decision Tree | 0.9371 | 0.9186 | 0.9551 | 0.9444 | 0.9497 | 0.8657
kNN | 0.979 | 0.9845 | 0.9677 | 1 | 0.9836 | 0.9555
Naive Bayes | 0.9371 | 0.9893 | 0.9263 | 0.9778 | 0.9514 | 0.865
Random Forest (Ensemble) | 0.958 | 0.995 | 0.9565 | 0.9778 | 0.967 | 0.9098
Support Vector Machine | 0.979 | 0.9969 | 0.9888 | 0.9778 | 0.9832 | 0.9553

E. Observations on Model Performance

Logistic Regression:
Performs very strongly on this dataset because the data is mostly separable after feature scaling. It achieves high AUC, recall, F1, and MCC.

Decision Tree:
Gives interpretable classification rules but performs slightly lower than the best models because a single tree can overfit or miss complex class boundaries.

kNN:
Performs well after standardization because distance-based models depend heavily on comparable feature scales.

Naive Bayes:
Provides a fast baseline and performs reasonably well, but its feature independence assumption limits peak performance.

Random Forest (Ensemble):
Performs strongly and is more stable than a single decision tree because it combines many trees and reduces variance.

Support Vector Machine:
Delivers highly competitive performance after scaling and is useful when class separation is strong in a transformed feature space.

Overall Winner for this dataset:
Logistic Regression is selected as the overall winner based on the highest F1 score, with MCC and AUC considered as supporting metrics.
