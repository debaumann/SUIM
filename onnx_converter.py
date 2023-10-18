import tensorflow as tf
import tf2onnx
import onnx 


model = tf.keras.models.load_model("models/suimnet_may.h5")
print(model.summary())

input_signature = [tf.TensorSpec([1, 240,320, 3], dtype=tf.float32)]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
onnx.save(onnx_model, "models/suimnet_may.onnx")