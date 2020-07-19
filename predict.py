from tensorflow.keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from parse import prediction_parser


def predict(image_path, model_path, classes_columns, n=3):
    model = load_model(model_path)
    img = img_to_array(
        load_img(path=image_path, target_size=(400, 400, 3)))/255.
    proba = model.predict(img.reshape(1, 400, 400, 3))
    top_n = np.argsort(proba[0])[:-(n+1):-1]
    for i in range(n):
        print("{}".format(classes_columns[top_n[i]]) +
              " ({:.3})".format(proba[0][top_n[i]]))
    plt.ion()
    plt.imshow(img)


if __name__ == "__main__":
    args = prediction_parser()
    classes_columns = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                       'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    print(predict(args.image, args.model, classes_columns, n=args.n))
