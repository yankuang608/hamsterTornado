#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess  = data['dsid']

        # return the inserted document ids
        documentId = self.db.labeledinstances.insert(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"documentId":str(documentId),
            "feature":[str(len(fvals))+" Points Received",
                    "min of: " +str(min(fvals)),
                    "max of: " +str(max(fvals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''

        dsid = self.get_int_arg("dsid", default=0)

        # create feature vectors from database
        f = [];
        for a in self.db.labeledinstances.find({"dsid": dsid}):
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l = [];
        for a in self.db.labeledinstances.find({"dsid": dsid}):
            l.append(a['label'])

        #specify parameter in each model

        #SVM
        df_function = "ovr"
        clf_SVC = SVC(gamma="scale", decision_function_shape=df_function)


        #Random Forest
        n_trees = 150
        clf_RF = RandomForestClassifier(n_estimators=n_trees, random_state = 0)


        #KNN
        n_neighbors = 1
        clf_KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

        # fit the model to the data
        accuracy = { "SVM": -1,
                "RandomForest": -1,
                "N_Neighbors": -1}

        models = {"SVM": clf_SVC,
                 "RandomForest": clf_RF,
                  "N_Neighbors": clf_KNN}

        if l:
            for model in models:
                clf = models[model]
                clf.fit(f,l)
                lstar = clf.predict(f)

                acc = sum(lstar == l) / float(len(l))
                accuracy[model] = acc
                bytes = pickle.dumps(clf)
                query = {"dsid": dsid}
                newvalues = {"$set": { model: Binary(bytes)}}
                self.db.models.update(query, newvalues, upsert=True)


        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json(accuracy)


class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        dsid = data['dsid']
        type = "RandomForest"

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!

        if(self.clf == []):
            print('Loading Model From DB')

            tmp = self.db.models.find_one({"dsid" : dsid})
            self.clf = pickle.loads(tmp["N_Neighbors"])

        predLabel = self.clf.predict(fvals);
        self.write_json({"prediction":str(predLabel)})
