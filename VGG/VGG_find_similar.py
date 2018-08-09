from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os


# consine similarity
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# convert all images to arrays
y_test = []
x_test = []
for img_path in os.listdir("images"):
    if img_path.endswith(".jpg"):
        img = image.load_img("images/" + img_path, target_size=(224, 224))
        y_test.append(int(img_path[0:2]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if len(x_test) > 0:
            x_test = np.concatenate((x_test, x))
        else:
            x_test = x

# convert input to VGG format
x_test = preprocess_input(x_test)

# include_top=False: exclude top(last) 3 fully-connected layers. get features dim=(1,7,7,512)
model = VGG16(weights='imagenet', include_top=False)

# use VGG to extract features
features = model.predict(x_test)

# flatten as one dimension
features_compress = features.reshape(len(y_test), 7 * 7 * 512)

# compute consine similarity
cos_sim = cosine_similarity(features_compress)

# random sampling 5 to test
inputNos = np.random.choice(len(y_test), 5, replace=False)

for inputNo in inputNos:
    top = np.argsort(-cos_sim[inputNo], axis=0)[1:3]
    recommend = [y_test[i] for i in top]
    output = 'input: \'{}\', recommend: {}'.format(inputNo + 1, recommend)
    print(output)
