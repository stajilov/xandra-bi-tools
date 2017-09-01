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
from sklearn.metrics.pairwise import linear_kernel

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


#xlsx loader
class XlsxLoader(Step):

    def __init__(self):
        self.datasetName = settings["xlsx_path"]
        self.sheetName = settings["sheet_name"]

    def execute(self, o):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        encoded_df = pd.read_excel(open(self.datasetName,'rb'), sheetname=self.sheetName)
        encoded_df = encoded_df.fillna(method='ffill')
        pprint(encoded_df.head(settings["rows_to_debug"]))
        pprint(encoded_df.shape)
        return encoded_df

#target document loader
class TargetDocumentLoader(Step):
    def __init__(self):
        self.documentDirectory = settings["documents_path"]
        self.documentsExtension = settings["documents_extension"]
        self.contentToWrite = settings["lookup_settings"]["target_document_text"]

    def execute(self, o):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        fileName = self.documentDirectory + "000000000000" + self.documentsExtension
        file = open(fileName, 'w+')
        file.write(self.contentToWrite)
        file.close()



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

        #pprint(df.head(settings["rows_to_debug"]))
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
        stems = [wordnet_lemmatizer.lemmatize(t) for t in filtered_tokens]
        #stems = [stemmer.stem(t) for t in filtered_tokens]

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


#lookup algorithms

class CosineSimilarityAlgorithm(Step):
    def __init__(self):
        self.params = settings["lookup_settings"]["similarity_params"]
        self.path = settings["documents_path"]
        #self.newColumn = settings["clustering_settings"]["target_column"]

    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        cosine_similarities = linear_kernel(df[0:1], df).flatten()
        print(cosine_similarities)
        numberOfDocsToFetch = self.params["target_doc_number"]
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        print(related_docs_indices)
        print(cosine_similarities[related_docs_indices])



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
        print(set(clusters))
        pprint(df.head(settings["rows_to_debug"]))
        df.to_csv(settings["df_dump_file_name"], index=False, encoding='utf-8')
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


#splitter
class TrainTestSplitter(Step):

    def __init__(self):
        self.trainTestRatio = settings["train_test_split_ratio"]
        self.targetColumn = settings["classification_regression_target"]

    def execute(self, df):
        y = df.pop(self.targetColumn)
        X = df
        X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)

        df_train = X.iloc[X_train]
        df_test = X.iloc[X_test]

        return df_train,df_test,y_train,y_test


#Feature inspect
class FeatureInspector():
    def __init__(self):
        self.trainTestRatio = settings["train_test_split_ratio"]

    def execute(self, clf):
        importances  = sorted(clf.feature_importances_)
        print("Number of features: ")
        print(len(importances))
        importantFeatures = list(filter(lambda x: x > 0.0, importances))
        if len(importantFeatures) < 10:
            print(importantFeatures)

        print("Number of important features: ")
        print(len(importantFeatures))


#classificaion algorithms

class ClassifierAlgorithm(Step):
    def trainTestSplitDataframe(self, df):
        ttSplitter = TrainTestSplitter()
        X_train,X_test,y_train,y_test = ttSplitter.execute(df)
        return X_train,X_test,y_train,y_test


class RFTClassifierAlgorithm(ClassifierAlgorithm):

    def __init__(self):
        self.params = settings["classification_settings"]["rftclassifier_params"]

    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        X_train,X_test,y_train,y_test = self.trainTestSplitDataframe(df)
        clf = RandomForestClassifier(**self.params)
        clf.fit(X_train, y_train)
        fInsp = FeatureInspector()
        fInsp.execute(clf)
        predictions = clf.predict(X_test)
        print("Accuracy is ", accuracy_score(y_test,predictions)*100,"%", "for params: ", self.params)

class SVMClassifierAlgorithm(ClassifierAlgorithm):
    def __init__(self):
        self.params = settings["classification_settings"]["svmclassifier_params"]

    def execute(self,df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        X_train,X_test,y_train,y_test = self.trainTestSplitDataframe(df)
        clf = svm.SVC(**self.params)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("Accuracy is ", accuracy_score(y_test,predictions)*100,"% for params:",self.params)


class KNearestNeighboursNClassifierAlgorithm(ClassifierAlgorithm):

    def __init___(self):
        self.params = settings["classification_settings"]["knnclassifier_params"]


    def execute(self, df):
        pprint(self.__class__.__name__)
        pprint(inspect.stack()[0][3])
        print(self.params)
        X_train,X_test,y_train,y_test = self.trainTestSplitDataframe(df)
        clf = KNeighborsClassifier(**self.params)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("Accuracy is ", accuracy_score(y_test,predictions)*100,"% for params:",self.params)

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
        self.algorithm = settings["classification_settings"]["algorithm"]

    def generate(self):
        if self.algorithm == "random_forest_trees":
            print("RFT")
            return RFTClassifierAlgorithm()
        elif self.algorithm == "knn":
            print("KNN")
            return KNearestNeighboursNClassifierAlgorithm()
        elif self.algorithm == "svm":
            print("SVM")
            return SVMClassifierAlgorithm()
        else:
            raise NotImplementedError




class RegressionFactory(AlgorithmAbstractFactory):
    def __init__(self):
        self.problem = settings["regression_settings"]["algorithm"]


class LookupFactory(AlgorithmAbstractFactory):
    def __init__(self):
        self.algorithm = settings["lookup_settings"]["algorithm"]

    def generate(self):
        if self.algorithm == "cosine_similarity":
            print("Cosine Similarity")
            return CosineSimilarityAlgorithm()
        else:
            raise NotImplementedError


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
        elif self.problem == "lookup":
            return LookupFactory()
        else:
            raise NotImplementedError




class XandraApp():

    def run(self):
        pipeline = Pipeline()

        s1 = CsvLoader()
        #s2 = DocumentLoader()
        #s1 = TargetDocumentLoader()
        s2 = ColumnsRemover()
        s3 = ColumnsEncoder()
        s4 = TfIdfProcessor()
        s5 = Purifier()
        s6 = ProblemFactory().generate().generate()

        #pipeline.addStep(s0)
        pipeline.addStep(s1)
        pipeline.addStep(s2)
        pipeline.addStep(s3)
        pipeline.addStep(s4)
        pipeline.addStep(s5)
        pipeline.addStep(s6)
        pipeline.executePipeline()

if __name__ == "__main__":
    app = XandraApp()
    app.run()
