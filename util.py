from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten

def split_data(data, test_size, shuffle=False):
    if shuffle == True:
        data = data.sample(frac=1).reset_index(drop=True)
    return data.iloc[0:int(len(data)*(1-test_size))], data.iloc[int(len(data)*(1-test_size)):]


def append_ext(name):
    return name+'.jpg'


def get_generators(data, images_dir, augment=False, test_size=0.1, shuffle=True, seed=42, train_batch_size=32, test_batch_size=8):
    # Train Validation Split
    training, validation = split_data(
        data, test_size=test_size, shuffle=shuffle)
    classes_columns = data.columns[2:].tolist()
    # Generators with Train data augmentation
    val_gen = ImageDataGenerator(rescale=1./255.)
    if augment:
        train_gen = ImageDataGenerator(rescale=1./255., rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                       horizontal_flip=True, shear_range=0.2, zoom_range=0.3)
        train_generator = train_gen.flow_from_dataframe(dataframe=training, directory=images_dir, x_col="Id", y_col=classes_columns,
                                                        batch_size=train_batch_size, seed=seed, shuffle=True, class_mode="raw",
                                                        target_size=(400, 400))
    else:
        train_generator = val_gen.flow_from_dataframe(dataframe=training, directory=images_dir, x_col="Id", y_col=classes_columns,
                                                      batch_size=train_batch_size, seed=seed, shuffle=True, class_mode="raw",
                                                      target_size=(400, 400))
    val_generator = val_gen.flow_from_dataframe(dataframe=validation, directory=images_dir, x_col="Id", y_col=classes_columns,
                                                batch_size=test_batch_size, seed=seed, shuffle=True, class_mode="raw",
                                                target_size=(400, 400))
    return train_generator, val_generator

def model_design():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5),
                    activation="relu", input_shape=(400, 400, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5),
                    activation="relu", input_shape=(400, 400, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
    return model