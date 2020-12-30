import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import statistics
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from dython.nominal import associations
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from hyperopt import hp, tpe, fmin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, plot_precision_recall_curve, roc_auc_score

def load_and_check_data(path):

    data = pd.read_csv(path,
                       sep=',')
    print(data.shape)
    print(data.head())
    print(data.dtypes)
    print(data.columns)
    print("Successfully loaded data from CSV")
    return data

def feature_engineering(data):

    # Change TotalCharges to numeric data type
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
    print(data["TotalCharges"])

    # Add new column wether costumer has automatic payment or not
    data['AutomaticPayment'] = np.where(data['PaymentMethod'].str.contains('(automatic)'), 'Yes', 'No')
    print(data['AutomaticPayment'])

    # Change 0 and 1 in SeniorCitizen column to Yes and No 
    data["SeniorCitizen"].replace({0:"No", 1:"Yes"}, inplace=True)
    print(data["SeniorCitizen"])

    return data
    
def exploratory_data_analysis(data, categoricals, numericals, plot_with_target = False, plot_corr_mat = False, save = False):
    
    if plot_corr_mat:
        associations(data[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

    for cat in categoricals:
        sns.countplot(x = cat, data = data)
        plt.title("Distribution of " + str(cat))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.xlabel(None)
        if save:
            plt.savefig(fname=str(cat)+"_count.png")
        plt.show()

        if plot_with_target:
            if cat != 'Churn':
                splot = sns.countplot(x = cat, data = data, hue='Churn')
                plt.title("Distribution of " + str(cat) + " dependent to target variable")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.xlabel(None)
                for p in splot.patches:
                    height = p.get_height()
                    splot.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}'.format(height/float(len(data)) * 100) + "%",
                    ha="center") 
                if save:
                    plt.savefig(fname=str(cat)+"_count_target.png")
                plt.show()

    for num in numericals:
        
        sns.distplot(data[num])
        plt.tight_layout()
        plt.title("Distribution of " + str(num))
        plt.xlabel(None)
        if save:
            plt.savefig(fname=str(num)+"_dist.png")
        plt.show()

        if plot_with_target:
            data.groupby("Churn")[num].apply(lambda x: sns.distplot(x, label= x.name))
            plt.tight_layout()
            plt.title("Distribution of " + str(num) + " dependent to target variable")
            plt.xlabel(None)
            plt.legend()
            if save:
                plt.savefig(fname=str(num)+"_dist_target.png")
            plt.show()

        sns.boxplot(x = num, data = data)
        plt.tight_layout()
        plt.title(str(num) + " boxplot")
        plt.xlabel(None)
        if save:
            plt.savefig(fname=str(num)+"_box.png")
        plt.show()

def preprocess_data(data, numericals, to_drop, seed, test_size, scale = False, oversample = False):

    # Exclude CustomerID and target variable from features and define encoders
    X = data[[x for x in list(data.columns) if x not in ["customerID", "Churn"] and x not in to_drop]]
    y = data["Churn"]

    enc = OneHotEncoder(handle_unknown = 'error', drop='if_binary')
    le = LabelEncoder()
    sc = StandardScaler()
    over = RandomOverSampler(random_state=seed)

    # Split the data in training and test set using a stratified split as the dataset is unbalanced (roughly 25% of customers churned)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # Oversample minority class 
    if oversample:
        X_train, y_train = over.fit_resample(X_train, y_train)


    # Remove unwanted columns from numericals list
    numericals = [num for num in numericals if num not in to_drop]

    # Remove numerical columns from the data for one-hot encoding
    X_train_onehot = enc.fit(X_train.drop(numericals, axis=1)).transform(X_train.drop(numericals, axis=1)).toarray()
    X_train_onehot_names = enc.get_feature_names()
    print(X_train_onehot_names)

    # Z-transform numerical values
    if scale:
        X_train[numericals] = \
            sc.fit(X_train[numericals]).transform(X_train[numericals])
        X_test[numericals] = \
            sc.transform(X_test[numericals])
        
    # Concatenate one-hot encoded data with numerical data
    X_train = np.concatenate((X_train[numericals].to_numpy(),X_train_onehot), axis=1)
    print("Shape of training data: " + str(X_train.shape))

    # Repeat for test set
    X_test_onehot = enc.transform(X_test.drop(numericals, axis=1)).toarray()
    X_test = np.concatenate((X_test[numericals].to_numpy(),X_test_onehot), axis=1)
    print("Shape of test data: " + str(X_test.shape))

    # Encode target
    y_train = le.fit(y_train).transform(y_train)
    print("Shape of training target: " + str(y_train.shape))
    y_test = le.transform(y_test)
    print("Shape of test target: " + str(y_test.shape))

    return X_train, X_test, y_train, y_test

def tune_xgb(seed, scoring):
    
    #This function searches through hyperparameter space given a scoring function
    def objective_xgb(params):
        params = {'n_estimators': int(params['n_estimators']),
                  'max_depth': params['max_depth'],
                  'learning_rate': params['learning_rate'],
                  'gamma': params['gamma'],
                  'min_child_weight': params['min_child_weight'],
                  'subsample': params['subsample']}
        xgb_clf = xgb.XGBClassifier(**params) 
        best_score = cross_val_score(xgb_clf, X_train, y_train, scoring=scoring, cv=StratifiedKFold(5), n_jobs=2).mean()
        loss = 1 - best_score
        return loss

    #Space of hyperparameters to search in
    space_xgb = {'n_estimators' : hp.quniform('n_estimators', 100,300,25),
                'max_depth' : hp.choice('max_depth', np.arange(1, 11, dtype=int)),
                'learning_rate': hp.uniform('learning_rate', 0.001, 0.9),
                'gamma' : hp.choice('gamma', np.arange(1, 11, dtype=int)),
                'min_child_weight' : hp.choice('min_child_weight', np.arange(1, 11, dtype=int)),
                'subsample': hp.uniform('subsample', 0.1, 1.0)}

    #Built-in fmin function finds best hyperparameters
    best_xgb = fmin(fn=objective_xgb,space=space_xgb, max_evals=20, rstate=np.random.RandomState(seed), algo=tpe.suggest)
   
    #n_estimators are returned as floats and converted to integers for further usage
    best_xgb["n_estimators"] = int(best_xgb["n_estimators"])

    print(best_xgb)
    return best_xgb

def evaluate_model(model, X_train, X_test, y_train, y_test, cv, scoring, seed, save = False):

    # Cross-validate model on training data to estimate performance
    cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
    print("Mean training balanced accuracy after CV: " + str(cv_results.mean()))

    # Fit the model to training data and evaluate on test data
    model.fit(X_train, y_train)
    y_true, y_pred= y_test, model.predict(X_test)
    
    # Balanced accuracy because of imbalanced dataset
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    class_rep = classification_report(y_true, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("Classification Report of test set: " + str(class_rep))
    print("Balanced accuracy on test set: " + str(bal_acc))
    print("ROC AUC score of test set: " + str(roc_auc))

    # Precision/Recall curved is favored in imbalanced classification tasks
    plot_precision_recall_curve(model, X_test, y_test) 
    plt.title('Precision/Recall curve')
    if save:
        plt.savefig(fname="PrecisionRecallCurve.png")
    plt.show()   
    
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm,
                 annot=True,
                 fmt="d",
                 cbar=False,
                 cmap="Blues",
                 xticklabels=['No Churn', 'Churn'],
                 yticklabels=['No Churn', 'Churn'])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix test set')
    if save:
        plt.savefig(fname="ConfusionMatrix.png")
    plt.show()

# Set random seed, load the dataset and check its shape and datatypes
random_seed = 191
data = load_and_check_data('TelcoData.csv')
print(list(data.columns))

# Check for missing values, remove them as there are only 11 and check for succesaful removal again
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.isnull().sum())

# Check mean, std etc and check target distribution
print(data.describe())
print(data['Churn'].value_counts())

# Feature Engineering
feature_engineering(data)

# Define lists for the different plots in the EDA
cats = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", \
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", \
    "Contract", "PaperlessBilling", "AutomaticPayment", "Churn"]
nums = ["tenure","MonthlyCharges","TotalCharges"]

# EDA
#exploratory_data_analysis(data, cats, nums, True, True, False)

# Define variables that are to be dropped. Create train and test set with one-hot encoded features and encoded target.
to_drop = ["gender"]
X_train, X_test, y_train, y_test = preprocess_data(data, nums, to_drop, random_seed, 0.25, scale=True, oversample=False)

# Define CV 
cv = RepeatedStratifiedKFold(n_splits=5 ,n_repeats=2)

# Tune model hyperparameters
best = tune_xgb(random_seed, 'balanced_accuracy')
optimized_model = xgb.XGBClassifier(**best,random_state=random_seed)
model = xgb.XGBClassifier(random_state=random_seed)

# Evaluate optimized model
evaluate_model(optimized_model, X_train, X_test, y_train, y_test, cv, 'balanced_accuracy', random_seed, True)
