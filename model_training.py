print('importing packages')
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# get training data
train = pd.read_csv("./data/training_data.csv")
# drop customer ID: not a feature for training
train.drop("customerID", axis=1, inplace=True)

# getting validation data
val = pd.read_csv("./data/validation_data.csv")

"""
Data Pre-Processing
The following columns need to be fixed in order to train the model:
1. All yes/no categories category encoded
2. All category columns category encoded
3. "Churn" column category encoded
"""

categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn']
# converting all the categorical columns to numeric
col_mapper = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(train.loc[:, col])
    class_names = le.classes_
    train.loc[:, col] = le.transform(train.loc[:, col])
    # saving encoder for each column to be able to inverse-transform later
    col_mapper.update({col: le})

train.replace(" ", "0", inplace=True)

# converting "Total Charges" to numeric
train.loc[:, "TotalCharges"] = pd.to_numeric(train.loc[:, "TotalCharges"])

# converting data pre-processing steps to a function to apply to new data
def pre_process_data(df, label_encoder_dict):
    df.drop("customerID", axis=1, inplace=True)
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            column_le = label_encoder_dict[col]
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue

    return df

# splitting into X and Y
x_train = train.drop("Churn", axis=1)
y_train = train.loc[:, "Churn"]

# fitting model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# pre-processing validation data
val = pre_process_data(val, col_mapper)

# split validation set
x_val = val.drop("Churn", axis=1)
y_val = val.loc[:, "Churn"]

# predicting on validation
predictions = model.predict(x_val)
precision, recall, fscore, support = precision_recall_fscore_support(y_val, predictions)
accuracy = accuracy_score(y_val, predictions)
print(f"Validation accuracy is: {round(accuracy, 3)}")

# pickling mdl
mdl_pickler = open("churn_prediction_model.pkl", "wb")
pickle.dump(model, mdl_pickler)
mdl_pickler.close()

# pickling le dict
le_pickler = open("churn_prediction_label_encoders.pkl", "wb")
pickle.dump(col_mapper, le_pickler)
le_pickler.close()

print('done')