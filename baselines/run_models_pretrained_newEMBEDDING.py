import _pickle as cPickle
#from keras.models import load_model
import numpy as np
import csv
#from models import *
from models_small_pretrained_train import *
from feature_extraction import train_test_features, save
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib



#dumpFeatures(False, False, True, None, 'features_pretrained_small.dat')
#[X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small.dat", "rb"))
[X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small_newEMBEDDING.dat", "rb"))
#model = build_model2(max_features+1, W.shape[1], X_train.shape[1], embeddings=W, trainable=True)


# keras_model = KerasClassifier(build_fn=build_model1, verbose=0)
# param_grid = {
#     'epochs': [2, 3],
#     'batch_size': [64, 128],
#     'dropout_rate': [0, 0.2, 0.5]
# }

# grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, verbose=11)
# grid_result = grid.fit(X_train, y)

# # Summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# #joblib.dump(grid.best_estimator_, 'features_pretrained_small/features_pretrained_small_MODEL.pkl')
# cPickle.dump(grid_result.best_params_, open('features_pretrained_small/features_pretrained_small_MODEL1_best_params_.pkl', 'wb'))
# cPickle.dump(grid_result.cv_results_, open('features_pretrained_small/features_pretrained_small_MODEL1_cv_results_.pkl', 'wb'))
# #cPickle.dump(grid_result.best_estimator_, open('features_pretrained_small_MODEL_cv_results_.pkl', 'wb'))
# grid_result.best_estimator_.model.save("features_pretrained_small/features_pretrained_small_MODEL1_best_estimator_")





# keras_model = KerasClassifier(build_fn=build_model2, verbose=0)
# param_grid = {
#     'epochs': [2, 3],
#     'batch_size': [64, 128],
#     'kernel_size': [3, 5],
#     'dropout_rate': [0, 0.2, 0.5]
# }

# grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, verbose=11)
# grid_result = grid.fit(X_train, y)

# # Summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# #joblib.dump(grid.best_estimator_, 'features_pretrained_small/features_pretrained_small_MODEL.pkl')
# cPickle.dump(grid_result.best_params_, open('features_pretrained_small/features_pretrained_small_MODEL2_best_params_.pkl', 'wb'))
# cPickle.dump(grid_result.cv_results_, open('features_pretrained_small/features_pretrained_small_MODEL2_cv_results_.pkl', 'wb'))
# #cPickle.dump(grid_result.best_estimator_, open('features_pretrained_small_MODEL_cv_results_.pkl', 'wb'))
# grid_result.best_estimator_.model.save("features_pretrained_small/features_pretrained_small_MODEL2_best_estimator_")






#keras_model = KerasClassifier(build_fn=build_model, verbose=0)
#keras_model = KerasClassifier(build_fn=model, verbose=0)
keras_model = KerasClassifier(build_fn=build_model3, verbose=0)
param_grid = {
    'epochs': [3],
    'batch_size': [128],
    'kernel_size': [3, 5],
    #'dropout_W': [0, 0.2, 0.5],
    #'dropout_U': [0, 0.2, 0.5],
    'filters': [16, 32],
    'units': [50, 100, 150]
}

grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, verbose=11)
grid_result = grid.fit(X_train, y)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#joblib.dump(grid.best_estimator_, 'features_pretrained_small/features_pretrained_small_MODEL.pkl')
cPickle.dump(grid_result.best_params_, open('features_pretrained_small/features_pretrained_small_MODEL3_best_params_.pkl', 'wb'))
cPickle.dump(grid_result.cv_results_, open('features_pretrained_small/features_pretrained_small_MODEL3_cv_results_.pkl', 'wb'))
#cPickle.dump(grid_result.best_estimator_, open('features_pretrained_small_MODEL_cv_results_.pkl', 'wb'))
grid_result.best_estimator_.model.save("features_pretrained_small/features_pretrained_small_MODEL3_best_estimator_")






# keras_model = KerasClassifier(build_fn=build_model4, verbose=0)
# param_grid = {
#     'epochs': [2, 3],
#     'batch_size': [64, 128],
#     'kernel_size': [3, 5],
#     'dropout_W': [0, 0.2, 0.5],
#     'dropout_U': [0, 0.2, 0.5],
#     'units': [50, 100, 150]
# }


# grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, verbose=11)
# grid_result = grid.fit(X_train, y)

# # Summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# #joblib.dump(grid.best_estimator_, 'features_pretrained_small/features_pretrained_small_MODEL.pkl')
# cPickle.dump(grid_result.best_params_, open('features_pretrained_small/features_pretrained_small_MODEL4_best_params_.pkl', 'wb'))
# cPickle.dump(grid_result.cv_results_, open('features_pretrained_small/features_pretrained_small_MODEL4_cv_results_.pkl', 'wb'))
# #cPickle.dump(grid_result.best_estimator_, open('features_pretrained_small_MODEL_cv_results_.pkl', 'wb'))
# grid_result.best_estimator_.model.save("features_pretrained_small/features_pretrained_small_MODEL4_best_estimator_")


