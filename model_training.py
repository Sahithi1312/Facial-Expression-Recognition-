import sys
sys.path.append('D:/ML/emotiondetector/venv/CNN_Model')

import numpy as np

from keras.callbacks import LambdaCallback, EarlyStopping

# Import model file

import CNN_Model.CNN as vc

def main():
    model = vc.CNN_16()

    X_fname = 'X_train_train.npy'
    y_fname = 'y_train_train.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    print(X_train.shape)
    print(y_train.shape)
   
    print("Training started")

    callbacks = []
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
    epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
    callbacks.append(earlystop_callback)
    callbacks.append(batch_print_callback)
    callbacks.append(epoch_print_callback)

    batch_size = 512
    model.fit(X_train, y_train, nb_epoch=400, \
            batch_size=batch_size, \
            validation_split=0.2, \
            shuffle=True, verbose=0, \
            callbacks=callbacks)

    model.save_weights('my_model_weights.h5')
    scores = model.evaluate(X_train, y_train, verbose=0)
    print ("Train loss : %.3f" % scores[0])
    print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")

if __name__ == "__main__":
    main()
