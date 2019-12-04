from keras.models import load_model
import pandas as pd

# load model
model = load_model('model.h5')

test  = pd.read_csv("test.csv").values

testX = test[:,0:].reshape(test.shape[0],1, 28, 28).astype( 'float32' )
X_test = testX / 255.0

my_prediction = model.predict_classes(X_test)

submission_mnist = pd.DataFrame({"ImageId" : list(range(1, 28001)), "Label" : my_prediction})


#creating csv file
submission_mnist.to_csv("submission.csv", index = False)