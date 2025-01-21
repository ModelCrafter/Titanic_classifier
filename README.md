# README

## Project Overview
This project is a machine learning solution aimed at predicting passenger survival on the Titanic. Using the Titanic dataset, it demonstrates data preprocessing, feature engineering, and classification models to achieve accurate predictions.

The Titanic dataset originates from a Kaggle competition and is widely used for beginner to intermediate-level data science projects. It includes information about passengers such as age, gender, ticket class, and survival status. The primary objective is to build a model that predicts whether a passenger survived the Titanic disaster based on the available features.

---

## Dataset Overview
The Titanic dataset contains two main files:
1. **train.csv**: Includes labeled data (with the `Survived` column) used for model training.
2. **test.csv**: Includes unlabeled data used for predictions.

Each record represents a passenger, and key features include:
- `Survived`: Target variable indicating survival (1 = survived, 0 = did not survive).
- `Pclass`: Ticket class (1st, 2nd, 3rd).
- `Name`: Passenger's name.
- `Sex`: Gender.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings or spouses aboard.
- `Parch`: Number of parents or children aboard.
- `Ticket`: Ticket number.
- `Fare`: Ticket price.
- `Cabin`: Cabin number.
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Preprocessing and Feature Engineering
The preprocessing steps are designed to handle missing values, encode categorical variables, and normalize numerical features. The project also includes custom transformations to enhance predictive power.

### Key Steps:
1. **Handling Missing Data**:
   - Missing `Age` and `Fare` values are imputed with their mean.
   - Missing categorical values like `Embarked` are imputed with the most frequent category.

2. **Dropping Irrelevant Features**:
   - Features such as `Name`, `Ticket`, and `PassengerId` are removed as they do not contribute significantly to prediction.

3. **Custom Features**:
   - `Sum_P_S`: Sum of `SibSp` and `Parch` to represent total family size aboard.
   - `Divion_S_P`: Ratio of `SibSp` to `Parch` to identify relationship structures.
   - `multi_P_S`: Product of `SibSp` and `Parch` to capture interaction effects.

4. **Clustering-Based Features**:
   - A custom transformer clusters passengers based on selected features and computes their similarity to cluster centers using an RBF kernel. These features aim to group passengers with similar survival probabilities.

---

## Models and Hyperparameter Tuning
Various classification models were tested using `GridSearchCV` for hyperparameter tuning. The models and their best accuracy scores during cross-validation include:

1. **Gradient Boosting Classifier**:
   - Best Parameters: `n_estimators=400`, `learning_rate=0.03`, `max_depth=5`, `loss='log_loss'`
   - Accuracy: **83.0%**

2. **Support Vector Classifier (SVC)**:
   - Best Parameters: `C=2`, `gamma=0.2`, `kernel='rbf'`
   - Accuracy: **83.8%**

3. **Random Forest Classifier**:
   - Best Parameters: `max_features=6`, `n_estimators=80`
   - Accuracy: **82.1%**

4. **K-Nearest Neighbors Classifier (KNN)**:
   - Best Parameters: `n_neighbors=10`, `weights='uniform'`
   - Accuracy: **80.9%**

---

## Results and Insights
- The **SVC** model achieved the highest accuracy of 83.8%, making it the best-performing model for this dataset.
- Feature engineering, especially clustering-based features, contributed to improved model performance.
- Hyperparameter tuning played a crucial role in optimizing model performance.

---

## Conclusion and Future Work
This project demonstrates the application of machine learning pipelines to a real-world dataset. While the models perform well, further improvements can include:
- Exploring deep learning models for survival prediction.
- Conducting feature importance analysis to refine the feature set.
- Testing the approach on similar datasets to assess generalizability.

---

## License
This project is licensed under the Apache License 2.0.

---
## Developer
***Youssef khaled***

Thank you for exploring this project! If you have any feedback or questions, feel free to reach out.

