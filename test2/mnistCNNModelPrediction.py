# Importing the Keras libraries and packages
from keras.models import load_model
model = load_model('tmp/model.ckpt.meta')

from PIL import Image
import numpy as np
from scipy import misc

for index in range(1,8):
    img = Image.open('c_' + str(index) + '.png').convert("L")
    # img = misc.imresize(img, 1.2)
    img = img.resize((28,28))
    im2arr = np.array(img)
    print(im2arr.shape)
    im2arr = im2arr.reshape(1,28,28,1)
    # Predicting the Test set results
    y_pred = model.predict(im2arr)

    for i in range(1):
		print(y_pred)