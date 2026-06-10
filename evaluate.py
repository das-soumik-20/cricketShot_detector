from preprocess import preprocess_data
import model
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = [' drive', 'legglance-flick', 'pullshot', 'sweep']


model = tf.keras.models.load_model("Cricket_Shot_Detection_model.keras")

_,_,test_ds = preprocess_data()

print("Test Set Evalauation-")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy - {test_accuracy*100:.2f}%")

#confusion matrix
y_true, y_pred = [],[]
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap='Blues')
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches='tight')

print("--- Classification Report ---")
print(classification_report(
    y_true, 
    y_pred, 
    labels=[0, 1, 2, 3],        # Maps all four class positions explicitly
    target_names=CLASS_NAMES,
    zero_division=0             # Handles edge cases where a class has 0 samples gracefully
))

