import tensorflow as tf
import numpy as np

CLASS_NAMES = ["drive", "legglance-flick", "pullshot", "sweep"]

model = tf.keras.models.load_model("Cricket_Shot_Detection_model.keras")

path = "test/smith_sweep.webp"
img = tf.keras.utils.load_img(path, target_size= (224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis= 0)

preds = model.predict(img_array)[0]
idx = np.argmax(preds)
print(f"Predicted shot: {CLASS_NAMES[idx]} with probability of {float(preds[idx])*100:.2f}%")