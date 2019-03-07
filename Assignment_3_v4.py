#!/usr/bin/env python
# coding: utf-8
# Yu-Jui Chen

from __future__ import print_function

import sys
from pyspark.sql import SparkSession

#set up functions for tokenize, top5 words, anagrams, bi-grams
def tokenize(x):
    word_list = x.lower()
    word_list = word_list.split()
    #strip special characters
    word_list = [x.strip("-\_\,\.\?\;\/\%\!\#\^\@\&\'\(\)\[\]\:\{\}") for x in word_list]
    #make sure that word starts an English alphabet and ends with an English alphabet 
    word_list2 = []
    for x in word_list:
        if len(x) < 2:
            if x.isalpha():
                word_list2.append(x)
        else:
            if (x[0].isalpha()) & (x[-1].isalpha()):
                                   word_list2.append(x)    
    return word_list2 

def Top5words(x):
    frequency = []
    if len(x) >= 4:
        frequency.append((x, 1))
    return frequency

def anagrams(x):
    anagrams_list = []
    #sort characters in a word and join them again
    if (len(x) >= 4):
        anagrams_list.append((''.join(sorted(x)), 1))
    return anagrams_list

def bigrams(x):
    #zip words in two lists and create the bigram words with turple 
    bi_list = list(zip(x, x[1:]))
    bi_list = [(x, 1) for x in bi_list]
    return bi_list

if __name__ == "__main__":
    input_path = sys.argv[1]
    spark = SparkSession.builder.appName("Assignment_3_Yu-Jui").getOrCreate()
    #set the log level for output
    spark.sparkContext.setLogLevel("ERROR") 
    myrdd = spark.sparkContext.textFile(input_path)
    Top5words_output = myrdd.flatMap(tokenize)\
                            .flatMap(Top5words)\
                            .reduceByKey(lambda x, y: x + y)\
                            .takeOrdered(5, key = lambda x: -x[1])
    anagrams_output = myrdd.flatMap(tokenize)\
                           .flatMap(anagrams)\
                           .reduceByKey(lambda x, y: x + y)\
                           .takeOrdered(5, key = lambda x: -x[1])
    bigrams_output = myrdd.map(tokenize)\
                          .flatMap(bigrams)\
                          .reduceByKey(lambda x, y: x + y)\
                          .takeOrdered(5, key = lambda x: -x[1])

    print('Top-5 words: ',  Top5words_output)
    print('Top-5 anagrams: ', anagrams_output)
    print('Top-5 bigrams: ', bigrams_output)

spark.stop()    

