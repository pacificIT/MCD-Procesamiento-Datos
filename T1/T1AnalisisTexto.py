import collections
import itertools
import math
import multiprocessing
from nis import match  
import string
from time import time
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 

class TFIDFMapReduce (object):

    def __init__(self , map_func, reduce_func, files, stopwords, num_workers=None):
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.InputsFiles = files
        self.Stopwords = stopwords
        self.pool = multiprocessing.Pool(num_workers)

    def WordMapping (self, mapped_values):
        mapping_data = collections.defaultdict(list)

        for value in mapped_values:
            countVectorizer = CountVectorizer()
            countVectorizer.stop_words =  nltk.corpus.stopwords.words('english')  

            countWordsOnDoc = countVectorizer.fit_transform(value)

            i=0 
            for word, v  in countVectorizer.vocabulary_.items():
                vd = countWordsOnDoc.data[i]
                mapping_data[word].append(vd)
                i=i+1
           
        return mapping_data.items()

    def __call__(self, inputs, chunksize=1):

        map_responses = self.pool.map(self.map_func, inputs, chunksize=chunksize)
        mapping_data = self.WordMapping(map_responses)
      #  print(mapping_data)
        reduced_Values = self.pool.map(self.reduce_func, mapping_data)

        return reduced_Values


def ReadFileToText (filename):

    print ( multiprocessing.current_process().name,'reading', filename)
    with open(filename) as f:
          lines = f.readlines()

    return lines
        
def CountWords(item):

    word, occurances = item
    nDocuments =  len(occurances)
    totaloccurances = sum(occurances) 

    return(word, totaloccurances, nDocuments)

# TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)
# tf(t,d) = count of t in d / number of words in d
# idf(t) = log(N/(df + 1))
def CalculateIDF(Word_counts, TotalWordsInDocs, TotalDocs):
    resltList = collections.defaultdict(object)

    for word, occurances, inDocs in Word_counts:
        TF        = occurances / TotalWordsInDocs
        #Frecuencia de documento inversa (IDF) = log (número total de documentos en el corpus / número de documentos que contienen la palabra + 1)
        IDF       = math.log ( TotalDocs / ( inDocs + 1 )  )
        TFIDF     = TF*IDF 
        resulRow  = {'Occurances': occurances,
                     'Documents': inDocs, 
                     'TF' : TF, 
                     'IDF' : IDF , 
                     'TF-IDF': TFIDF}
                     
        resltList[word] = resulRow

    return resltList

def GetOccurancesFromAllDocs(dict):
     sum = 0
     for  word, occurances, inDocs in dict:
           sum = sum + occurances
     return sum
 

if __name__ == '__main__':
    import operator
    import glob

    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
 
    #input_files = glob.glob('**/*.txt', recursive=True)
    input_files = glob.glob('4280-1.txt')

    t1 = time()
    tfidfMapReduce  = TFIDFMapReduce(ReadFileToText, CountWords, input_files, stopwords, 6)
    word_counts     = tfidfMapReduce(input_files)

    TotalOccurances = GetOccurancesFromAllDocs(word_counts)
    TotalDocs       =  len(input_files)

    word_counts.sort(key=operator.itemgetter(1))
    word_counts.reverse()
    #print(word_counts)

    TFIDFs          =  CalculateIDF(word_counts, TotalOccurances, TotalDocs)
    df              = pd.DataFrame.from_dict(TFIDFs, orient='index')
    datatoexcel     = pd.ExcelWriter('Resultado-TF_IDF.xlsx')
    df.to_excel(datatoexcel)
    datatoexcel.save()

    lm_time         = time() - t1

    print ('top de palabras ordenados por frecuencia')
    topWords = word_counts[:10]
    longest = max(len(word) for word, count, d in topWords)
    for word, count, d in topWords:
        print ('%-*s: %5s   %5s' % (longest+1, word, count, d))

    print(lm_time)
