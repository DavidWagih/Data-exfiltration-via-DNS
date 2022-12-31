
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD

class Model_trainer():

    """
    A class used to train several classifier models

    ...

    Attributes
    ----------
    model : str
        model name to use one of the following supported:
        RF  : RandomForestClassifier
        KNN : KNeighborsClassifier
        GB  : GradientBoostingClassifier
        SVC : Support Vector Classifier
        DNN : Keras Deep Neural Network
    """

    
    def __init__(self,model):
        self.selected_model=model


    def fit(self,X_train,y_train):
        if "RF" in str(self.selected_model):
            param_grid = {
                'bootstrap': [True],
                'max_depth': range(3,8),
                'max_features': [2, 3 ,4],
                'n_estimators': [100, 200, 300, 400]
            }

            #rf= RandomForestClassifier()
            #grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = 4, verbose = 2)                         
            #grid_search.fit(X_train,y_train)

            #grid_search.best_params_
            #grid_search.best_score_

            best_params = {'bootstrap': True, 'max_depth': 7, 'max_features': 4, 'n_estimators': 100} # final output parameters from GridSearch shown above in comments


            rf= RandomForestClassifier(**best_params)
            rf.fit(X_train,y_train)
            self.selected_model=rf

        elif "GB" in str(self.selected_model):
            param_grid = {
                'max_features': range(3,8),
                'loss': ('deviance', 'exponential'),
                'learning_rate': [0.1,0.01,0.001,0.0001,0.00001],
                'n_estimators': [100, 200],
                'random_state':[0]
            }

            #gb= GradientBoostingClassifier()
            #grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, cv = 3, n_jobs = 4, verbose = 2)
            #grid_search.fit(X_train,y_train)

            best_params = {'learning_rate': 0.1, 'loss': 'deviance', 'max_features': 7, 'n_estimators': 200, 'random_state': 0} # final output parameters from GridSearch
    

            gb= GradientBoostingClassifier(**best_params)
            gb.fit(X_train,y_train)
            self.selected_model=gb

        elif "KNN" in str(self.selected_model):
    
            knn = KNeighborsClassifier(5)
            knn.fit(X_train,y_train)
            self.selected_model=knn

        elif "SVC" in str(self.selected_model):
            svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            svc.fit(X_train, y_train)
            self.selected_model=svc


        elif "DNN" in str(self.selected_model):
            # define the keras model
            model = Sequential()
            model.add(Dense(128, input_dim=12, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            # compile the keras model
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
            # fit the keras model on the dataset
            model.fit(X_train, y_train, epochs=1, batch_size=256,validation_split=0.1)

            self.selected_model=model

    def predict(self,X_test):
        return self.selected_model.predict(X_test)