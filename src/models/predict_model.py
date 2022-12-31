
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import numpy as np

class Model_predict():
    def __init__(self,model):
        self.selected_model=model


    def predict(self,X_test,y_test):
        try:
            acc= accuracy_score(y_test,self.selected_model.predict(X_test))
            f1 = f1_score(y_test,self.selected_model.predict(X_test))
            print("Accuracy Score over test set :",acc)
            print("F1 Score over test set :",f1)
            print("Confusion Matrix :\n",confusion_matrix(y_test,self.selected_model.predict(X_test)))
        except:
            y_pred = np.where(self.selected_model.predict(X_test)>0.5,1,0)
            acc= accuracy_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            print("Accuracy Score over test set :",acc)
            print("F1 Score over test set :",f1)
            print("Confusion Matrix :\n",confusion_matrix(y_test,y_pred))

        return acc,f1
        