import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
def data_split(data,ratio):
    shuffled= np.random.permutation(len(data))
    test_set_size=int(len(data)* ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df=pd.read_csv("H:\\IDP-6months\\Machine Learning\\Coronavirus_predictor\\Flask Web App\\accuracy_check1.csv")
    train,test=data_split(df,0.2)
    X_train=train[["Age","Fever","Fatigue","Runny_Nose","Difficulty_in_breath"]].to_numpy()
    X_test=test[["Age","Fever","Fatigue","Runny_Nose","Difficulty_in_breath"]].to_numpy()
    Y_train=train[["Infection_probability"]].to_numpy().reshape(4000,)
    Y_test=test[["Infection_probability"]].to_numpy().reshape(999,)
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)