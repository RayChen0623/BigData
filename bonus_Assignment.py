#!/usr/bin/env python
# coding: utf-8
# Yu-Jui Chen

from __future__ import print_function

import sys
import pandas as pd
import pyspark
from numpy import array
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
import re

# Create a function for removing stopword, punctuation, special charaters 
def stopword_punctuation(x):
    words = []
    for i in x:
        if i not in stopwords.words('english'):
            words.append(re.sub('[^A-Za-z0-9]+', '',i))
    return words

def spam(x):
    if (x[1] == 1.0):
        return (x[0],'Spam')
    else:
        return (x[0],'NoSpam')

if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("Bonus Assignment").getOrCreate()

    Nospam_path = sys.argv[1] # No Spam Email Data
    Spam_path = sys.argv[2] # Spam Email Data
    input_data = sys.argv[3] # Predict Data 

    sc = spark.sparkContext

    # Load Data
    myrdd_NoSpam = sc.textFile(Nospam_path)
    myrdd_Spam = sc.textFile(Spam_path)
    myrdd_input = sc.textFile(input_data)

    # data cleaning: lowercase, remove stopword and punctuation
    ham_words = myrdd_NoSpam.map(lambda x: x.lower().split()).map(stopword_punctuation)
    spam_words = myrdd_Spam.map(lambda x: x.lower().split()).map(stopword_punctuation)
    input_words = myrdd_input.map(lambda x: x.lower().split('\t'))\
                    .map(lambda x: (stopword_punctuation(x[1].split()), x[0], x[1]))
    # create feature vectors and hashing them to a length 1000 feature vector 
    tf = HashingTF(numFeatures = 1000)

    # Map features
    spam_features = tf.transform(spam_words)
    ham_features = tf.transform(ham_words)
    input_features = input_words.map(lambda x: (tf.transform(x[1]), x[1]))

    # Label spam data as 1 
    spam_samples = spam_features.map(lambda x: LabeledPoint(1, x))
    ham_samples = ham_features.map(lambda x: LabeledPoint(0, x))
    input_samples = input_features.map(lambda x: ((LabeledPoint(0, x[0])),x[1]))

    # Union spam and no spam data for training model
    data = spam_samples.union(ham_samples)
    
    #log_res = LogisticRegressionWithSGD() #Choose the NaiveBayes as the final model
    #model = log_res.train(data)
    #results = input_samples.map(lambda x: (model.predict(x[0].features),x[1]))
    model2 = NaiveBayes.train(data)
    results_2 = input_samples.map(lambda x: (x[1], model2.predict(x[0].features)))


    print (results_2.map(spam).collect())

spark.stop()  




