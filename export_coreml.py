#!/usr/bin/python
'''Read from PyMongo, make simple model and export for CoreML'''

# make this work nice when support for python 3 releases
# from __future__ import print_function # python 3 is good to go!!!

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import numpy as np

# export 
import coremltools
import pickle



dsid = 36

client  = MongoClient(serverSelectionTimeoutMS=50)
db = client.sklearndatabase

document = db.models.find_one({"dsid":dsid})

clf_rf = pickle.loads(document["RandomForest"])
clf_svm = pickle.loads(document["SVM"])
clf_knn = pickle.loads(document["N_Neighbors"])

coreml_model = coremltools.converters.sklearn.convert(
	clf_rf) 

# save out as a file
coreml_model.save('RandomForest.mlmodel')


coreml_model = coremltools.converters.sklearn.convert(
	clf_svm) 

# save out as a file
coreml_model.save('SVM.mlmodel')

coreml_model = coremltools.converters.sklearn.convert(
	clf_knn)

# save out as a file
coreml_model.save('KNN.mlmodel')

# close the mongo connection

client.close()







