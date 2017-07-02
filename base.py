from __future__ import print_function

import sysimport pandas as pdimport numpy as np
import nltk
import re
import json
import inspect


#import seaborn as sns


import glob
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer



from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier




from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold

from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.pipeline import Pipeline

from pprint import pprint
from time import time
import logging


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import GridSearchCV

import sys
from abc import ABC, abstractmethod

sys.dont_write_bytecode = True

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
wordnet_lemmatizer = WordNetLemmatizer()

settings = []
with open('settings.json') as data_file:
    settings = json.load(data_file)



class Pipeline:
	def __init__(self):
		self.steps = []
		self.currentObject = None

	def addStep(self, step):
		self.steps.append(step)

	def executePipeline(self):
		for step in self.steps:
			out = step.execute(self.currentObject)
			self.currentObject = out


class Step(ABC):
	@abstractmethod
	def execute(self):
		pass


class DataLoader(Step):

	def __init__(self):
		self.datasetName = settings["dataset_path"]
		self.datasetSeparator = settings["dataset_separator"]

	def execute(self, o):
		pprint(self.__class__.__name__)
		pprint(inspect.stack()[0][3])
		encoded_df = pd.read_csv(self.datasetName, sep=self.datasetSeparator)
		encoded_df = encoded_df.fillna(method='ffill')

		return encoded_df



class ColumnsRemover(Step):

	def __init__(self):
		self.columns = settings["columns_to_remove"]

	def execute(self, df):
		pprint(self.__class__.__name__)
		pprint(inspect.stack()[0][3])
		for c in self.columns:
			df.drop(c, axis=1, inplace=True)

		pprint(df.head(settings["rows_to_debug"]))
		return df



class ColumnsEncoder(Step):

	def __init__(self):
		self.columns = settings["columns_to_encode"]

	def execute(self, df):
		pprint(self.__class__.__name__)
		pprint(inspect.stack()[0][3])

		encoded = self.transform(df)
		pprint(encoded.head(settings["rows_to_debug"]))
		return encoded


	def transform(self,X):

   		output = X.copy()
   		if self.columns is not None:
   			for col in self.columns:
   				output[col] = LabelEncoder().fit_transform(output[col])
   		else:
   			for colname,col in output.iteritems():
   				output[colname] = LabelEncoder().fit_transform(col)
   		return output

	def fit_transform(self,X,y=None):
		return self.fit(X,y).transform(X)




class TfIdfProcessor(Step):

	def __init__(self):
		self.columns = settings["columns_to_do_tfidf"]


	def tokenize_and_stem(self,text):
		tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
		filtered_tokens = []
		for token in tokens:
			if re.search('[a-zA-Z]', token):
				filtered_tokens.append(token)
		#stems = [wordnet_lemmatizer.lemmatize(t) for t in filtered_tokens]
		stems = [stemmer.stem(t) for t in filtered_tokens]
		return stems


	def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
		tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
		filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
		for token in tokens:
			if re.search('[a-zA-Z]', token):
				filtered_tokens.append(token)

		return filtered_tokens



	def getTfIdfMatrixForDF(self, df):
		local_df = df
		tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=3000, min_df=0.0001, stop_words='english', use_idf=True, tokenizer=self.tokenize_and_stem, ngram_range=(1,1))

		for c in self.columns:
			print(c)
			valuesOfDF = local_df.pop(c).values
			X = tfidf_vectorizer.fit_transform(valuesOfDF.astype('U')).toarray()
		#print(tfidf_vectorizer.get_feature_names())
			for i, col in enumerate(tfidf_vectorizer.get_feature_names()):
				local_df[col] = X[:, i]

		return local_df

	def execute(self, df):
		pprint(self.__class__.__name__)
		pprint(inspect.stack()[0][3])

		transformed = self.getTfIdfMatrixForDF(df)
		pprint(transformed.head(settings["rows_to_debug"]))

		return transformed

class Purifier(Step):

    def __init__(self):
        self.shouldPurify = settings["should_purify"]

    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])

        if(self.shouldPurify):
            local_df = df.fillna(method='ffill')
            local_df = local_df.rename(columns = {'fit': 'fit_feature'})
            pprint(local_df.head(settings["rows_to_debug"]))
            return local_df
        else:
            return df




pipeline = Pipeline()

s1 = DataLoader()
s2 = ColumnsRemover()
s3 = ColumnsEncoder()
s4 = TfIdfProcessor()
s5 = Purifier()

pipeline.addStep(s1)
pipeline.addStep(s2)
pipeline.addStep(s3)
pipeline.addStep(s4)
pipeline.addStep(s5)

pipeline.executePipeline()
