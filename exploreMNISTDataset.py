from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import json

# Load the hand written Dataset
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()

# Pick a sample to plot
sample = 1200

# Reshape image into a single row
image = X_train[sample]
image_array = image.reshape((1,784))
# Normalize the dataset
image_array_decimal = image_array/255
# Convert to JSON
lists = image_array_decimal.tolist()
json_str = json.dumps(lists)
print(json_str)

# Display the image
plt.imshow(image, cmap='gray')
plt.title(f"Label: {Y_train[sample]}")
plt.axis('off')
plt.show()