import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score, classification_report, confusion_matrix, roc_auc_score

# Classifiers
classifiers = [
    #('SVM', SVC(probability=True)),
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('RUSBoostClassifier', RUSBoostClassifier()),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('Random Forest', RandomForestClassifier())
]

# Parameter grid
param_grid = {
    'SVM': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'classifier__degree': [2, 3, 4],  # for poly kernel
        'classifier__random_state': [42]
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200, 500],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__random_state': [42]
    },
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2', 'elasticnet'],
        'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'classifier__random_state': [42]
    },
    'Decision Tree': {
        'classifier__max_depth': [None, 5, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__random_state': [42]
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    },
    'RUSBoostClassifier':{
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1],
        'classifier__algorithm': ['SAMME', 'SAMME.R'],
        'classifier__replacement': [True],
        'classifier__random_state': [42]
    }, 
    'AdaBoostClassifier':{
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 1],
        'classifier__algorithm': ['SAMME', 'SAMME.R'],
        'classifier__random_state': [42]
    }
}

"""
Function to perform grid search on each of the classification models to perform hyperparameter tuning

Params: 
    classifier: The classification model
    prprocessor: The preprocessor from the pipeline
    X_train: X train data split
    y_train: y train data split
    param_grid: Hyperparameters for each model
    use_smote: Boolean value to toggle the use of synthetic minority oversampling. False when using RUSBoostClassifier,
    BalancedRandomForestClassifier, and SMOTEBoostClassifier as these already handle imbalanced labels

# Returns:
    best_estimator_: The best estimator for each of the classification models
    
"""
def evaluate_model(classifier, preprocessor, X_train, y_train, param_grid, use_smote):
    smt = SMOTE(random_state=42)

    if use_smote is True:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smt', smt),
            ('classifier', classifier)
        ])
    elif use_smote is False:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

    grid_search = GridSearchCV(pipeline, param_grid, cv = 5, verbose = 3)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def evaluation_metrics(best_model, X_test, y_test, name, n_classes):
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Results for {name}:")
    print(classification_report(y_test, y_pred))
    print("Best parameters found:", best_model.get_params())
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-score: {f1 * 100:.2f}%')
    print("=" * 80)
    
    # Convert y_test to binary if it's not already
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    if y_test_bin.shape[1] == 1:  # Binary case
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plotting accuracy, precision, recall, and F1 score in a bar chart
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']  
    plt.figure(figsize=(10, 6))  
    bars = sns.barplot(x=metric_names, y=metrics)
    for p in bars.patches:
        bars.annotate(format(p.get_height(), '.2f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 10), 
                      textcoords = 'offset points')
    plt.title(f'Classification Metrics for {name}')
    plt.show()

    # Plotting Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Plotting ROC Curve for each class and micro-average
    plt.figure(figsize=(10, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

    # Micro-averaged AUC Score
    print(f"Micro-averaged AUC Score for {name}: {roc_auc['micro'] * 100:.2f}%")

"""
Function to run the pipeline by calling the preprocessing function, splitting the training and test set, applying SMOTE and 
performing grid search for hyperparameter tuning for each classifier and output the results.

Params:
    data: The dataset
    features: List of features to include when training the model
    test_size: Test size as a decimal between 0 and 1 (e.g. 0.2 would mean 20% test size and 80% train size)

Returns:
    Visualisation of evaluation metrics for the best estimator found by grid search for each classification algorithm
"""
def run_ml_pipeline(data, features, test_size):
    df = data
    
    # Preprocessing the data
    df.dropna(subset=features + ['Credit_Score'], inplace=True)

    # Splitting the features into numerical and categorical
    numerical_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object']).columns.tolist()

    # Preprocessor
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = df[features]
    y = df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y, test_size = test_size)

    best_auc = 0
    best_model = None
    best_model_name = ''

    # Evaluate and visualize metrics for each classifier
    for name, classifier in classifiers:
        classifier_param_grid = param_grid.get(name, {})
        use_smote = name not in ['RUSBoostClassifier']

        # current_best_model = evaluate_model(classifier, preprocessor, X_train, y_train, classifier_param_grid, use_smote)
        # #y_prob = current_best_model.predict_proba(X_test)[:, 1]
        # auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        
        # if auc_score > best_auc:
        #     best_auc = auc_score
        #     best_model = current_best_model
        #     best_model_name = name

        current_best_model = evaluate_model(classifier, preprocessor, X_train, y_train, classifier_param_grid, use_smote)
        y_prob = current_best_model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = current_best_model
            best_model_name = name
        
        n_classes = len(np.unique(y_train))
        evaluation_metrics(current_best_model, X_test, y_test, name, n_classes)

    print(f"Best model is {best_model_name} with AUC score: {best_auc}")

    return best_model, best_model_name, best_auc





    
    

