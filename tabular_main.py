import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
# Scikit-Learn algorithms
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


CSV_FOLDER = 'dataset/TabDatasets_Fahad/NIDS/'

base_learners = [
        DecisionTreeClassifier(),
        LinearDiscriminantAnalysis(),
    #LinearDiscriminant
        #XGBClassifier(),
    ]
multi_learners = [
        RandomForestClassifier(),
        XGBClassifier(),
        LogisticRegression(solver='sag'),
    ]

# TO_BE_REMOVE = ['OilLeak']
TO_BE_REMOVE = ['SSH-Bruteforce', 'FTP-BruteForce', 'Infilteration']
# TO_BE_REMOVE = ['[Arancino Service Stop]ProcessHangInjection(d5000-narancino)', '[Redis Generate Vars]RedisMemoryInjection(d5000)', '[RedisGet]RedisStressInjection(d5000-w10)', '[NodeRed Service Stop]ProcessHangInjection(d5000-nnode-red)']
pairs = list(product(base_learners, multi_learners))
# Iterating over CSV files in folder
for dataset_file in os.listdir(CSV_FOLDER):
    full_name = os.path.join(CSV_FOLDER, dataset_file)
    if full_name.endswith(".csv"):
        # if file is a CSV, it is assumed to be a dataset to be processed
        data = pd.read_csv(full_name, sep=",")
        # Split the data into features and target variable
        classes = data['multilabel'].unique()
        X = data.drop(columns=['multilabel'])
        y = data['multilabel'].apply(lambda x: 0 if x in TO_BE_REMOVE else 1)

        #Normal = 1 & Anomaly= 0


        # Remove the Values which are consider as anamolous in the Binary CLF
        df_multi_clf = data[y != 0]
        # df_filtered = data[data['multilabel'].apply(lambda x: x not in TO_BE_REMOVE)]
        df_multi_clf['multilabel'] = df_multi_clf['multilabel'].apply(lambda x: 'normal' if x == 'normal' else 'anomaly')

        data['binCLFlabel'] = y
        data['multiCLFlabel'] = df_multi_clf['multilabel']

        # data.to_csv(full_name+"_update.csv")



        label_encoder = LabelEncoder()

        X_multi = df_multi_clf.drop(columns=['multilabel'])
        y_multi = df_multi_clf['multilabel'].apply(lambda x: 1 if x == 'normal' else 0)



        #Binary CLF Splitting to Training and Testing
        # X = X.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        y_test_multi = data['multiCLFlabel'].iloc[X_test.index]
        y_test_multi = y_test_multi.apply(lambda x: 1 if x == 'normal' else 0)


        #Multi CLF Splitting to Training and Testing

        # X_multi = X_multi.to_numpy()
        X_train_multi, X_test_multi, y_train_multi, y_test_multip = train_test_split(X_multi, y_multi, test_size=0.3, random_state=42)



        # y_test_bin = y_test.apply(lambda x: 0 if x in TO_BE_REMOVE else 1)
        # y_train_bin = y_train.apply(lambda x: 0 if x in TO_BE_REMOVE else 1)

        # label_encoder = LabelEncoder()
        #
        # y_test_mul = label_encoder.fit_transform(y_test)
        # y_train_mul = label_encoder.fit_transform(y_train)

        print("Total Training Data Size of Binary CLF: ", len(X))
        print("Total Training Data Size of Multi CLF: ", len(X_multi))

        for classifier in pairs:
            print("=======================================================")
            print(f'Binary Classifier: {classifier[0].__class__.__name__}')
            print(f'Multilabel Classifier: {classifier[1].__class__.__name__}')
            print("=======================================================")

            bin_filename = CSV_FOLDER + "trained_model/binaryCLF/" + classifier[0].__class__.__name__ + ".joblib"
            multi_filename = CSV_FOLDER + "trained_model/" + classifier[1].__class__.__name__ + ".joblib"

            print(f'Binary Model: {classifier[0].__class__.__name__} saved in {bin_filename}')
            print(f'Multilabel Model: {classifier[1].__class__.__name__} saved in {multi_filename}')
            print('##########################################################################')
            #Binary CLF
            # Train a DecisionTreeClassifier
            bin_model = classifier[0]
            bin_model.fit(X_train, y_train)
            # Save the binary model using joblib
            dump(bin_model, bin_filename)
            # Load the model
            loaded_model = load(bin_filename)
            # Predict on the test set
            y_pred_bin = loaded_model.predict(X_test)


            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred_bin)
            print("Binary CLF Accuracy:", accuracy)


            # Train a MultiClass CLF
            multi_model = classifier[1]
            multi_model.fit(X_train_multi, y_train_multi)
            # Save the model using joblib
            dump(multi_model, multi_filename)

            # Load the model
            loaded_model = load(multi_filename)

            # Predict on the test set
            # y_pred = loaded_model.predict(X_test)
            y_pred = loaded_model.predict(X_train_multi)

            # y_pred_proba = loaded_model.predict_proba(X_test)
            y_pred_proba = loaded_model.predict_proba(X_train_multi)
            max_probabilities = np.max(y_pred_proba, axis=1)
            # Evaluate the model
            accuracy = accuracy_score(y_train_multi, y_pred)
            print("Multilabel Accuracy:", accuracy)

            #Save the results to DataFrame
            df = pd.DataFrame()
            # df['True Label'] = y_test
            # df['Predicted Label'] = y_pred_bin
            df['True CLF Labels'] = y_train_multi
            df['Predicted CLF'] = y_pred
            df['Probability'] = max_probabilities

            df.to_csv(classifier[0].__class__.__name__+"_"+classifier[1].__class__.__name__+'.csv', index = False)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")