from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, PReLU, Softmax, ReLU, Flatten, Dense

def get_pnet():
        inp = Input(shape=(12, 12, 3))

        layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inp)
        layer = PReLU(shared_axes=[1, 2])(layer)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(layer)
        layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(layer)
        layer = PReLU(shared_axes=[1, 2])(layer)
        layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(layer)
        layer = PReLU(shared_axes=[1, 2])(layer)

        layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), name='BB_REGRESSION')(layer)
        
        layer_out1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(layer)
        layer_out1 = Softmax(name='FACE_CLASSIFIER')(layer_out1)

        p_net = Model(inp, [layer_out2, layer_out1], name="PNet")

        return p_net

def get_rnet():
    inp = Input(shape=(24, 24, 3))

    layer = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inp)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(layer)
    layer = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid")(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(layer)
    layer = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid")(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = Flatten()(layer)
    layer = Dense(128)(layer)
    layer = PReLU()(layer)

    layer_out1 = Dense(2)(layer)
    layer_out1 = Softmax(axis=1)(layer_out1)

    layer_out2 = Dense(4)(layer)

    r_net = Model(inp, [layer_out2, layer_out1], name="RNet")

    return r_net

def get_onet():
    inp = Input(shape=(48, 48, 3))

    layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(inp)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(layer)
    layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(layer)
    layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(layer)
    layer = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="valid")(layer)
    layer = PReLU(shared_axes=[1, 2])(layer)
    layer = Flatten()(layer)
    layer = Dense(256)(layer)
    layer = PReLU()(layer)

    layer_out1 = Dense(2)(layer)
    layer_out1 = Softmax(axis=1, name='FACE_CLASSIFIER')(layer_out1)

    layer_out2 = Dense(4, name='BB_REGRESSION')(layer)

    o_net = Model(inp, [layer_out2, layer_out1], name="ONet")
    return o_net
