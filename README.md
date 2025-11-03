🧠 Stroke Prediction using Machine Learning

Reproduction Study of “A Comparative Analysis of Machine Learning Classifiers for Stroke Prediction: A Predictive Analytics Approach”

⸻

📘 Project Overview

This repository reproduces and analyzes the experiments from the published article:

A Comparative Analysis of Machine Learning Classifiers for Stroke Prediction: A Predictive Analytics Approach.

The study aims to evaluate the performance of multiple machine learning classifiers for stroke prediction using the Kaggle Stroke Prediction Dataset.
This work replicates the original paper’s approach and compares results obtained from this reproduction.

⸻

🎯 Objectives
	•	Reproduce the machine learning pipeline described in the published article.
	•	Compare model performance (Accuracy, Precision, Recall, F1, ROC-AUC).
	•	Analyze performance gaps between reproduced and published results.
	•	Interpret the models using SHAP feature importance.

⸻

🧩 Dataset Information

Source: Kaggle - Stroke Prediction Dataset￼
Size: 5,110 samples, 11 features + target (stroke).
Target Variable:
	•	stroke = 1 → Patient has had a stroke.
	•	stroke = 0 → Patient has not had a stroke.

Main Features:
gender, age, hypertension, heart_disease, ever_married,
work_type, Residence_type, avg_glucose_level, bmi, smoking_status.

⚙️ Installation & Setup

1️⃣ Clone or download this repository
git clone https://github.com/<yourusername>/stroke-prediction-ML.git
cd stroke-prediction-ML

2️⃣ Create and activate environment (optional but recommended)
python -m venv stroke_env
source stroke_env/bin/activate      # for Mac/Linux
stroke_env\Scripts\activate         # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧮 Dependencies

The following Python packages are required:
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
xgboost
shap

🧾 How to Run

In Google Colab
	1.	Open Google Colab￼.
	2.	Upload the file Final_exam_assignment.ipynb.
	3.	Upload the dataset stroke.csv or link it from Kaggle.
	4.	Run all cells sequentially (Runtime > Run all).

  🧠 Models Implemented
No   Model                  Description
1    Logistic Regression    Baseline linear model
2    Decision Tree          Single tree classifier
3    Random Forest          Bagging ensemble of decision trees
4    Gradient Boosting      Sequential ensemble learning
5    AdaBoost               Boosting technique focusing on errors
6    SVM                    Kernel-based classifier
7    KNN                    Distance-based classifier
8    MLP                    Multilayer Perceptron neural network
9    Naive Bayes            Probabilistic classifier
10   Nearest Centroid (NCC) Prototype-based classifier
11   Voting Classifier      Ensemble of multiple base models

Each model is tuned using GridSearchCV (10-fold cross-validation).

📊 Evaluation Metrics
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-Score
	•	ROC-AUC

Evaluation results are automatically printed in the notebook output.

⸻

📈 Results Summary (Your Reproduced Results)

Model                Accuracy    Precision    Recall    F1    ROC-AUC
Random Forest        0.941       0.27         0.03      0.05  0.79
Gradient Boosting    0.921       0.24         0.19      0.21  0.74
Logistic Regression  0.743       0.14         0.73      0.24  0.83
AdaBoost             0.742       0.15         0.74      0.24  0.83
Voting Classifier    0.847       0.17         0.46      0.25  0.82
… (see notebook for full table)


⸻

🧩 Interpretability

Feature importance is analyzed using SHAP (SHapley Additive Explanations).
Top important features include:
	•	age
	•	avg_glucose_level
	•	bmi
	•	hypertension
	•	heart_disease

⸻

📚 Outputs

After running all notebook cells, you will see:
	•	Data preprocessing summary
	•	Confusion matrices and ROC curves for each model
	•	SHAP summary and feature plots
	•	Final comparison table of all models

⸻

📘 Reference

Original Paper:

A Comparative Analysis of Machine Learning Classifiers for Stroke Prediction: A Predictive Analytics Approach
Healthcare Analytics, 2022.

⸻

✍️ Author

Your Name: Kyaw Min Htike
Institution: Khon Kaen University
Course: Machine Learning Final Examination Project
Date: November 2025

⸻

🧩 License

This reproduction is for academic and educational purposes only.
All rights to the original paper and dataset belong to their respective authors.
