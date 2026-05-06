from tensorflow.keras import layers, models
from resnet_model import build_resnet_base
from capsule_layer import CapsuleLayer

def build_hybrid_model(num_classes):

    resnet = build_resnet_base()

    inputs = layers.Input(shape=(224, 224, 3))

    x = resnet(inputs)

    # Dense bridge layer
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Capsule-style classification
    x = CapsuleLayer(num_capsules=num_classes, dim_capsule=16)(x)

    outputs = layers.Softmax()(x)

    model = models.Model(inputs, outputs, name="ResNet_CapsNet_Hybrid")

    return model
