from __future__ import print_function

import sysimport pandas as pdimport numpy as np
import nltk
import re
import json
import inspect
import glob

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

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

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


#csv loader
class CsvLoader(Step):

	def __init__(self):
		self.datasetName = settings["csv_path"]
		self.datasetSeparator = settings["csv_separator"]

	def execute(self, o):
		pprint(self.__class__.__name__)
		pprint(inspect.stack()[0][3])
		encoded_df = pd.read_csv(self.datasetName, sep=self.datasetSeparator)
		encoded_df = encoded_df.fillna(method='ffill')
		return encoded_df

#document loader
class DocumentLoader(Step):
    def __init__(self):
        self.documentDirectory = settings["documents_path"]
        self.documentsExtension = settings["documents_extension"]
        self.path = self.documentDirectory + "*" +  self.documentsExtension
        self.columnName = settings["text_column_name"]


    def execute(self, o):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        files = glob.glob(self.path)
        df = pd.DataFrame(columns=[self.columnName])
        for i in range(len(files)):
            file = files[i]
            f = open(file, 'r')
            content = f.read()
            #print(content)
            df.loc[i] = content
            f.close()

        pprint(df.head(settings["rows_to_debug"]))
        return df



#encoders, tranformators, cleaners, removers
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
        tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=3000, min_df=1, stop_words='english',
        use_idf=True, tokenizer=self.tokenize_and_stem, ngram_range=(1,1))
        for c in self.columns:
            print(c)
            valuesOfDF = local_df.pop(c).values
            print(valuesOfDF)
            X = tfidf_vectorizer.fit_transform(valuesOfDF.astype('U')).toarray()
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



#clustering algorithms
class KMeansAlgorithm(Step):
    def __init__(self):
        self.params = settings["clustering_settings"]["kmeans_params"]
        self.newColumn = settings["clustering_settings"]["target_column"]

    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])

        km = KMeans(**self.params)
        km.fit(df)
        clusters = km.labels_.tolist()
        df[self.newColumn] = clusters
        pprint(df.head(settings["rows_to_debug"]))
        return df

class HierarchicalAlgorithm(Step):
    def __init__(self):
        self.params = settings["clustering_settings"]["hierachical_params"]
        self.newColumn = settings["clustering_settings"]["target_column"]

    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])

        hierarchical = AgglomerativeClustering(**self.params)
        hierarchical.fit(df)
        clusters = hierarchical.labels_.tolist()
        df[self.newColumn] = clusters
        pprint(df.head(settings["rows_to_debug"]))
        return df

class DBScanAlgorithm(Step):
    def __init__(self):
        self.params = settings["clustering_settings"]["dbscan_params"]
        self.newColumn = settings["clustering_settings"]["target_column"]

    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])

        #dbScan = DBSCAN()
        #dbScan.fit(df)
        loc_df = StandardScaler().fit_transform(df)
        db = DBSCAN(**self.params).fit(loc_df)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        #labels = db.labels_
        clusters = db.labels_.tolist()
        print(clusters)


        loc_df[self.newColumn] = clusters
        pprint(df.head(settings["rows_to_debug"]))

        return loc_df



#algorithm factories
class AlgorithmAbstractFactory(ABC):
	@abstractmethod
	def generate(self):
		pass


class ClusteringFactory(AlgorithmAbstractFactory):
    def __init__(self):
        self.algorithm = settings["clustering_settings"]["algorithm"]

    def generate(self):
        if self.algorithm == "kmeans":
            print("K Means")
            return KMeansAlgorithm()
        elif self.algorithm == "dbscan":
            return DBScanAlgorithm()
        elif self.algorithm == "hierarchical":
            return HierarchicalAlgorithm()
        else:
            raise NotImplementedError

class ClassificationFactory(AlgorithmAbstractFactory):
    def __init__(self):
        self.problem = settings["classification_settings"]["algorithm"]



class RegressionFactory(AlgorithmAbstractFactory):
    def __init__(self):
        self.problem = settings["regression_settings"]["algorithm"]


class ProblemFactory():
    def __init__(self):
        self.problem = settings["problem"]

    def generate(self):
        if self.problem == "clustering":
            print("Clustering problem")
            return ClusteringFactory()
        elif self.problem == "classification":
            return ClassificationFactory()
        elif self.problem == "regression":
            return RegressionFactory()
        else:
            raise NotImplementedError




class XandraApp():

    def run(self):
        pipeline = Pipeline()

        s1 = DocumentLoader()
        #s2 = ColumnsRemover()
        #s3 = ColumnsEncoder()
        s4 = TfIdfProcessor()
        s5 = Purifier()
        s6 = ProblemFactory().generate().generate()

        pipeline.addStep(s1)
        #pipeline.addStep(s2)
        #pipeline.addStep(s3)
        pipeline.addStep(s4)
        pipeline.addStep(s5)
        pipeline.addStep(s6)
        pipeline.executePipeline()

if __name__ == "__main__":
    app = XandraApp()
    app.run()
