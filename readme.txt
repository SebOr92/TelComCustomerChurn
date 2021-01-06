Telcom Customer Churn

In this project I used the TelCom dataset provided by Kaggle to predict churn behaviour by using Extreme Gradient Boosting.

This readme contains a step by step description of the project.

First, the data is loaded to as a pandas dataframe and some inital information is printed to the console.

In the next step, the data type of "TotalCharges" is changed to numeric, as it is wrongly encoded in the original dataset.
The column "SeniorCitizen" is changed to Yes and No instead of 1 and 0 to match other binary columns.
As a last step of the feature engineering process, an additional column is added to check whether a customer has automatic payment or not.
This is done by using simple regular expression, from which the new column is derived.

Afterwards, the features are inspected in the exploratory data analysis. The resulting plots are included in this repository. 
Key take away are:
- the dataset is imbalanced
- there is no difference in the churn behaviour of female or male customers
- there are customers with a tenure of about 70 months that haven't churned yet
- the higher the monthly charges, the more people tend to churn
- customers without a partner are more likely to churn
For the EDA, the features are divided into numerical and categorical features. Depending on their type, different kinds of plots are created.

In the fourth step, the data is prepared to be used in a machine learning algorithm. First, the customerID and the target variable are dropped.
The customerID is not a feature, as it does not contain any information about a customer. If there are any other variables to be dropped, this 
happend here as well.
Encoders, scalers and oversamplers are defined at this stage too. I used the OneHotEncoder instead of the get_dummies function of pandas as it
is not suitable for usage in the actual preparing of the data for the learning task. Get_dummies is very likely to cause data leakage as there is
no way of handling unseen data in the test set that didn't appear in the training set. scikit-learns OHC handles these cases automatically. Also, if
the features are binary, the first category is dropped.
Next, the data is split into a training and test set using a stratified split because of the imbalance in the dataset. Optionally, the training data can
be oversampled at this stage. In the remaining steps, the categorical data is encoded, numerical features can be scaled optionally, and finally the
data is being merged together again and returned as numpy arrays.

Before the evaluation, another function to optimize hyperparameters is defined using the hyperopt library.

In the last step the model is evaluated by using different metrics. First, the performance on the training set is evaluated by using the mean of a
10-fold stratified cross validation with a given scoring function. Next, the model predicts the test data and is evaluated using balanced accuracy, 
ROC curve, ROC AUC score and a confusion matrix. 

The best model was able to score a balanced accuracy score of about 71.5% on the test set with a ROC AUC score of 0.86. The score of 71.3% on the training
data indicates, that the model doesn't overfit.
The confusion matrix shows, that the model does well at finding not-churning customers, but has problem with finding churing customers. This is 
further supported by the Precision/Recall curve.
This behaviour is due to the imbalance in the dataset and could further be tackled by trying more sampling techniques or changing the scoring 
function in the hyperparameter optimization process.
Depending on the worse of type of error, the parameters could be tuned again using for example F0.5 or F2 scores, depending on the importance of error types.

The results where achieved by dropping the "gender" column, z-transforming numerical data and optimizing the hyperparameters.

