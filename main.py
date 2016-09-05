####The entire data manipulation was stolen from the script from Dune_dweller
##https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/a-linear-model-on-apps-and-labels
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, merge, Input, MaxoutDense
from keras.layers.advanced_activations import ELU, PReLU
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import svm
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import os
from scipy.sparse import csr_matrix, hstack

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

datadir = 'C:\\Users\\nikhil\\Documents\\TalkingData'
gatrain = pd.read_csv(os.path.join(datadir, 'gender_age_train.csv'), index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir, 'gender_age_test.csv'), index_col='device_id')
phone = pd.read_csv(os.path.join(datadir, 'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id', keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir, 'events.csv'), parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir, 'app_events.csv'), usecols=['event_id', 'app_id', 'is_active'],
                        dtype={'is_active': bool})
applabels = pd.read_csv(os.path.join(datadir, 'app_labels.csv'))
labelcat = pd.read_csv(os.path.join(datadir, 'label_categories.csv'))

####Phone brand
# As preparation I create two columns that show which train or test set row a particular device_id belongs to.

gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

Xtr_1 = np.zeros((gatrain.shape[0], 1))
Xte_1 = np.zeros((gatest.shape[0], 1))
# device_ID features
train_deviceid = np.zeros((gatrain.shape[0], 20))
test_deviceid = np.zeros((gatest.shape[0], 20))
for index, data in gatrain.iterrows():
    s = "%.20d" % abs(index)
    Xtr_1[data['trainrow']] = float(data['trainrow']) / gatrain.shape[0]
    for j in range(0, 20):
        train_deviceid[data['trainrow']][j] = int(s[j])
    if index < 0:
        train_deviceid[data['trainrow']][0] = 1
        # print(str(train_deviceid[data['trainrow']]))
for index, data in gatest.iterrows():
    s = "%.20d" % abs(index)
    Xte_1[data['testrow']] = float(data['testrow']) / gatest.shape[0]
    for j in range(0, 20):
        test_deviceid[data['testrow']][j] = int(s[j])
    if index < 0:
        test_deviceid[data['testrow']][0] = 1


brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),
                        (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]),
                        (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))

# Device model
# In [5]:
m = phone.phone_brand.str.cat(phone.device_model)

modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]),
                        (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]),
                        (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left', left_on='event_id', right_index=True)
              .groupby(['device_id', 'app'])['app'].agg(['size'])
              .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
              .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
              .reset_index())

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)),
                     shape=(gatrain.shape[0], napps))
Xtr_totalapps = Xtr_app.sum(axis=1)
Xtr_totalapps = Xtr_totalapps / Xtr_totalapps.max()

d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)),
                     shape=(gatest.shape[0], napps))
Xte_totalapps = Xte_app.sum(axis=1)
Xte_totalapps = Xte_totalapps / Xte_totalapps.max()
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)

devicelabels = (deviceapps[['device_id', 'app']]
                .merge(applabels[['app', 'label']])
                .groupby(['device_id', 'label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)),
                       shape=(gatrain.shape[0], nlabels))
Xtr_totallabels = Xtr_label.sum(axis=1)
Xtr_totallabels = Xtr_totallabels / Xtr_totallabels.max()
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)),
                       shape=(gatest.shape[0], nlabels))
Xte_totallabels = Xte_label.sum(axis=1)
Xte_totallabels = Xte_totallabels / Xte_totallabels.max()

print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


labelcat = labelcat.loc[labelcat.label_id.isin(applabels.label_id.unique())]
labelcat['label'] = labelencoder.transform(labelcat.label_id)
categoryencoder = LabelEncoder().fit(labelcat.category)
labelcat['category'] = categoryencoder.transform(labelcat.category)
ncategories = len(categoryencoder.classes_)

devicecategories = (devicelabels[['device_id', 'label']]
                    .merge(labelcat[['label', 'category']])
                    .groupby(['device_id', 'category'])['label'].agg(['size'])
                    .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                    .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                    .reset_index())
devicecategories.head()

d = devicecategories.dropna(subset=['trainrow'])
Xtr_category = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.category)),
                          shape=(gatrain.shape[0], ncategories))
Xtr_totalcategories = Xtr_category.sum(axis=1)
Xtr_totalcategories = Xtr_totalcategories / Xtr_totalcategories.max()
d = devicecategories.dropna(subset=['testrow'])
Xte_category = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.category)),
                          shape=(gatest.shape[0], ncategories))
Xte_totalcategories = Xte_category.sum(axis=1)
Xte_totalcategories = Xte_totalcategories / Xte_totalcategories.max()
print('Categories data: train shape {}, test shape {}'.format(Xtr_category.shape, Xte_category.shape))

# Concatenate all features
Xtrain = hstack((Xtr_1, Xtr_model, Xtr_brand, Xtr_label, Xtr_app, Xtr_category), format='csr')
Xtest = hstack((Xte_1, Xte_model, Xte_brand, Xte_label, Xte_app, Xte_category), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)
dummy_y = np_utils.to_categorical(y)


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


def nnet():
    inputs = Input(shape=(Xtrain.shape[1],))

    l0 = Dense(100, init='normal')(inputs)
    l0 = PReLU()(l0)
    l0 = Dropout(0.4)(l0)

    l1 = Dense(100, init='normal')(l0)
    l1 = PReLU()(l1)
    l1 = Dropout(0.1)(l1)

    l1 = Dense(12, init='normal', activation='softmax')(l1)

    model = Model(input=inputs, output=l1)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model



def ensemble(X, Y, nfolds):
    # dataset_blend_train = np.zeros((X.shape[0], 12))
    num_fold = 0
    kf = KFold(X.shape[0], n_folds=nfolds, shuffle=True)
    for train_index, test_index in kf:
        num_fold += 1
        print('Start Fold: ' + str(num_fold))
        trainX = X[train_index]
        trainY = Y[train_index]
        testX = X[test_index]
        testY = Y[test_index]
        model = nnet()
        model.fit_generator(generator=batch_generator(trainX, trainY, 256, True),
                            nb_epoch=15,
                            samples_per_epoch=67840,
                            verbose=2
                            )
        save_model(model, num_fold, 'Arch')
        # dataset_blend_train[test_index, :] = model.predict_generator(generator=batch_generatorp(testX, 128, False), val_samples=testX.shape[0])
        # logit_clf.fit(dataset_blend_train, np_utils.probas_to_classes(Y))


def vote(limit, Xtest):
    results = np.zeros((Xtest.shape[0], 12))
    for i in range(limit):
        model = read_model(i + 1, 'Arch')
        scores = model.predict_generator(generator=batch_generatorp(Xtest, 128, False), val_samples=Xtest.shape[0])
        results += scores
    results /= limit
    return results


# model = nnet()

X_train, X_val, y_train, y_val = train_test_split(Xtrain, dummy_y, test_size=0.002, random_state=42)

ensemble(Xtrain, dummy_y, 20)
scores_val = vote(20, X_val)
scores = vote(20, Xtest)

'''
#2nd stacked neural network
scores_train = vote(15, X_train)
# Mix results with top level features
scores_train = np.concatenate((X_train[:, :3].toarray(), scores_train), axis=1)
scores_val = np.concatenate((X_val[:, :3].toarray(), scores_val), axis=1)
scores = np.concatenate((Xtest[:, :3].toarray(), scores), axis=1)
def nnet2():
    inputs = Input(shape=(scores_train.shape[1],))

    l0 = Dense(50, init='normal')(inputs)
    l0 = PReLU()(l0)
    l0 = Dropout(0.4)(l0)

    l1 = Dense(50, init='normal')(l0)
    l1 = PReLU()(l1)
    l1 = Dropout(0.1)(l1)

    l1 = Dense(12, init='normal', activation='softmax')(l1)

    model = Model(input=inputs, output=l1)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
model = nnet2()
model.fit(scores_train, y_train, batch_size=128, nb_epoch=20, validation_data=(scores_val, y_val))
scores = model.predict(scores)

fit = model.fit_generator(generator=batch_generator(X_train, y_train, 100, True),
                          nb_epoch=15,
                          samples_per_epoch=73600,
                          validation_data=(X_val.todense(), y_val), verbose=2,
                          )

# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 64, False), val_samples=X_val.shape[0])
scores = model.predict_generator(generator=batch_generatorp(Xtest, 64, False), val_samples=Xtest.shape[0])
'''
print('logloss val {}'.format(log_loss(y_val, scores_val)))

pred = pd.DataFrame(scores, index=gatest.index, columns=targetencoder.classes_)
pred.to_csv('output.csv', index=True)
