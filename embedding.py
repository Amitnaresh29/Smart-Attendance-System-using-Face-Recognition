from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

class emb:
    def __init__(self):
        self.model=load_model("C:/Users/admin/Downloads/Smart-Attendance-System-using-Face-Recognition-master/Smart-Attendance-System-using-Face-Recognition-master/PreTrained_model/facenet_keras.h5")
     
    def calculate(self,img):
        return self.model.predict(img)[0]
