# Utility Libraries
import pandas as pd
from datetime import datetime
import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
from util import append_ext, get_generators, model_design
from parse import training_parser
K.clear_session()

if __name__=="__main__":
    # Parsing Arguments
    args = training_parser()

    # Constants
    SEED = args.seed
    IMAGES_PATH = args.images
    CSV_PATH = args.csv
    LOG_DIR = args.logs
    MODEL_PATH = args.ckpt
    MODEL_CKPT_PATH = os.path.join(MODEL_PATH, 'poster-ckpt.hdf5')


    # Data
    data = pd.read_csv(CSV_PATH)

    # Appending Extension
    data['Id'] = data['Id'].apply(append_ext)
    classes_columns = data.columns[2:].tolist()
    print(data.head())

    # Get Data Generators
    train_generator, val_generator = get_generators(
        data, IMAGES_PATH, augment=False, test_size=0.1, shuffle=True, seed=SEED, train_batch_size=32, test_batch_size=8)

    # Model Design
    model = model_design()

    # Setting Model Callbacks
    log = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = TensorBoard(log, histogram_freq=1)
    ckpt = ModelCheckpoint(filepath=MODEL_CKPT_PATH,
                        verbose=1, save_best_only=True)

    # Fitting the Model
    history = model.fit(train_generator, epochs=2,
                        validation_data=val_generator, callbacks=[tb_callback, ckpt])

    # Saving The Model
    model.save(MODEL_PATH+str(datetime.now().strftime("%Y%m%d-%H%M%S")) +
            '-poster-classifier.hdf5')
