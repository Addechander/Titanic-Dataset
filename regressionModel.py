from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf       

dftest = pd.DataFrame({"survived": 0,
                    "sex": ["male"],
                   "age": [34],
                   "n_siblings_spouses": [4], 
                   "fare": [34.4], 
                   "class": ["First"], 
                   "deck": ["C"], 
                   "alone": ["y"]})


ytest = dftest.pop("survived")

def Reg(dataset, label):
    dftrain = pd.read_csv('Data/Titanic Dataset/eval.csv') # training data
    dfeval = pd.read_csv('Data/Titanic Dataset/eval.csv') # testing data  

    dftrain.pop("parch")
    dfeval.pop("parch")
    dftrain.pop("embark_town")
    dfeval.pop("embark_town")

    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')


    categorical_column =['sex', 'n_siblings_spouses', 'class', 'deck','alone']

    numeric_columns = ['age', 'fare']

    feature_columns = []

    #converting categorical data into numeric data 
    for feature_name in categorical_column:
        vocabulary = dftrain[feature_name].unique() #gets a list of all unique values from given feature
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in numeric_columns:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    
    def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
        def input_function():  # inner function, this will be returned
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
            if shuffle:
                ds = ds.shuffle(1000)  # randomize order of data
            ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
            return ds  # return a batch of the dataset
        return input_function  # return a function object for use

    train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    #Creating Model
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)  # train
    result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

    accuracy = result['accuracy']

    #User specified data

    #def input_fn(features, batch_size=264):
        #return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    
    predict_input_fn = eval_input_fn = make_input_fn(dataset, label, num_epochs=1, shuffle=False)
    
    prediction = list(linear_est.evaluate(predict_input_fn))
    return accuracy, prediction


accuracy, prediction = Reg(dftest, ytest)

print(accuracy, prediction)

