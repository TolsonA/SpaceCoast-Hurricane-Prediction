import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def logistic_regression_and_visualization(hu_binary_amo_df):
    """Perform logistic regression and visualize the results."""
    # Prepare the data for logistic regression
    X = hu_binary_amo_df['AMO_Anomaly']
    y = hu_binary_amo_df['Hurricanes']

    # Add a constant to the independent variables matrix
    X = sm.add_constant(X)

    # Fit the logistic regression model
    logit_model = sm.Logit(y, X).fit()

    # Print the summary of the model
    print(logit_model.summary())

    # Extract predictions
    predictions = logit_model.predict(X)

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='AMO_Anomaly', y='Hurricanes', data=hu_binary_amo_df, label='Hurricanes', alpha=0.5)
    sns.lineplot(x=hu_binary_amo_df['AMO_Anomaly'], y=predictions, color='red', label='Logistic Regression Line')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label='0-deg Line')
    plt.xlabel('AMO Anomaly')
    plt.ylabel('Hurricane Yes/No')
    plt.title('Logistic Regression of Hurricanes/AMO Anomaly')
    plt.legend(loc='center left')
    plt.show()

    return logit_model

def random_forest_classifier(hu_slp_df, amo_data):
    """Train and evaluate a Random Forest classifier."""
    # Merge SLP data with AMO data
    hu_slp_df = hu_slp_df.merge(amo_data, on='Year')

    X2 = hu_slp_df.drop(columns=['Hurricanes'])
    y2 = hu_slp_df['Hurricanes']

    # Check for consistency in the number of samples
    assert len(X2) == len(y2), "Inconsistent number of samples between X2 and y2"

    # Split the data
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.22, random_state=42)

    # Check for consistency after splitting
    assert len(X_train2) == len(y_train2), "Inconsistent number of samples between X_train2 and y_train2"
    assert len(X_test2) == len(y_test2), "Inconsistent number of samples between X_test2 and y_test2"

    # Initialize and train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=300,
                                           class_weight='balanced', 
                                           max_depth=10,
                                           min_samples_split=2,
                                           min_samples_leaf=3,
                                           max_features='sqrt')
    rf_classifier.fit(X_train2, y_train2)

    # Make predictions on the test set
    threshold = rf_classifier.predict_proba(X_test2)[:,1]
    y_pred_rf_classifier = (threshold >= 0.44).astype(int)
    #y_pred_rf_classifier = rf_classifier.predict(X_test2)

    # Evaluate the model
    accuracy = accuracy_score(y_test2, y_pred_rf_classifier)
    precision = precision_score(y_test2, y_pred_rf_classifier, average='weighted', zero_division=0)
    recall = recall_score(y_test2, y_pred_rf_classifier, average='weighted')
    f1 = f1_score(y_test2, y_pred_rf_classifier, average='weighted')
    conf_matrix = confusion_matrix(y_test2, y_pred_rf_classifier)

    return rf_classifier, accuracy, precision, recall, f1, conf_matrix