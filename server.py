import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn import datasets
import sys, os
from flask import Flask
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from flask import jsonify

app = Flask(__name__)
@app.route('/exam')
def predict():
    new_filenames = os.listdir('image')
    new_df = pd.DataFrame({'filename' : new_filenames})
    nb_samples = new_df.shape[0]
    test_datagen1 = ImageDataGenerator(rescale=1./255)
    test_generator1 = test_datagen1.flow_from_dataframe(new_df,'image',
                                                        x_col='filename', y_col='None', 
                                                        target_size=(150,150),
                                                        class_mode=None,
                                                        batch_size = 1, shuffle = False)

    new_model = load_model('perfume2.h5')
    predict = new_model.predict_generator(test_generator1,
                                  steps=np.ceil(nb_samples/2))
    new_df['category'] = np.argmax(predict, axis=-1)
    new_df['category'] = new_df['category'].replace({0:'warmc', 1: 'jimmy', 2: 'lanvi'})
    b=new_df.loc[0,'category']
    return jsonify(str(b))

if __name__== "__main__":
    print(("dfjdf"))
    app.run(host="0.0.0.0")
