import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from scipy.optimize import leastsq
import svgwrite
from PIL import Image
from svgpathtools import svg2paths, svg2paths2, Path
from io import BytesIO

# Paths to datasets
data_train_path = 'doodle_dataset/train'
data_test_path = 'doodle_dataset/test'
data_val_path = 'doodle_dataset/val'

# Image dimensions
img_width = 180
img_height = 180

# Load datasets
data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=None)

data_cat = data_train.class_names

data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=None)

data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=None)

plt.figure(figsize=(10, 10))
for images, labels in data_train.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(data_cat[labels[i].numpy()])
        plt.axis('off')
plt.show()

model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dense(len(data_cat))
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

epochs_size = 25
history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

model_save_path = 'doodle_classifier_model'
model.save(model_save_path)

epochs_range = range(epochs_size)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

image_path = 'path/to/image/to/predict.jpg'
image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image)
img_bat = tf.expand_dims(img_arr, 0)

prediction = model.predict(img_bat)
predicted_class = data_cat[np.argmax(prediction)]
print(f"Predicted class: {predicted_class}")

def fit_circle(XY):
    def calc_R(xc, yc):
        return np.sqrt((XY[:, 0] - xc) ** 2 + (XY[:, 1] - yc) ** 2)
    
    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    center_estimate = np.mean(XY[:, 0]), np.mean(XY[:, 1])
    center, _ = leastsq(f, center_estimate)
    Ri = calc_R(*center)
    R = Ri.mean()
    
    if np.std(Ri) < 0.1 * R:
        return np.array([center[0], center[1], R])
    else:
        return None

def fit_ellipse(XY):
    x = XY[:, 0]
    y = XY[:, 1]
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    if np.any(np.isnan(a)):
        return None
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b*b-a*c
    if num <= 0:
        return None
    cx = (c*d-b*f)/num
    cy = (a*f-b*d)/num
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*( ((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c))) )-(c+a))
    down2 = (b*b-a*c)*( ((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c))) )-(c+a))
    semi_major = np.sqrt(up/down1)
    semi_minor = np.sqrt(up/down2)
    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major
    theta = 0.5*np.arctan(2*b/(a-c))
    return cx, cy, semi_major, semi_minor, theta

def fit_rectangle(XY):
    min_x, min_y = np.min(XY, axis=0)
    max_x, max_y = np.max(XY, axis=0)
    if np.std(np.diff(XY[:, 0])) < 0.1 * (max_x - min_x) and np.std(np.diff(XY[:, 1])) < 0.1 * (max_y - min_y):
        return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    else:
        return None

def detect_and_regularize_shapes(paths_XYs):
    regularized_paths = []
    for path in paths_XYs:
        for XY in path:
            circle = fit_circle(XY)
            if circle is not None:
                regularized_paths.append(circle)
                continue
            ellipse = fit_ellipse(XY)
            if ellipse is not None:
                regularized_paths.append(ellipse)
                continue
            rect = fit_rectangle(XY)
            if rect is not None:
                regularized_paths.append(rect)
                continue
            regularized_paths.append(XY)
    return regularized_paths

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = 'blue'
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()
    
def save_to_csv(paths_XYs, csv_path):
    with open(csv_path, 'w') as f:
        for i, path in enumerate(paths_XYs):
            for j, XY in enumerate(path):
                line = f"{i},{j}," + ",".join(map(str, XY.flatten())) + "\n"
                f.write(line)

def process_data(input_file, output_file):
    data = np.genfromtxt(input_file, delimiter=',')
    with open(output_file, 'w') as f_out:
        for row in data:
            x1, y1, x2, y2 = row
            processed_x1 = x1
            processed_y1 = y1
            processed_x2 = x2 * 0.95
            processed_y2 = y2 * 0.95
            f_out.write(f"{processed_x1},{processed_y1},{processed_x2},{processed_y2}\n")

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'blue', 'green', 'yellow']
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()

    # Convert SVG to PNG using Pillow
    svg_image = Image.open(svg_path)
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    width, height = fact * W, fact * H
    png_image = svg_image.resize((width, height), Image.ANTIALIAS)
    png_image.save(png_path, 'PNG')

def main(input_csv_path, output_csv_path, processed_output_csv_path):
    process_data(input_csv_path, processed_output_csv_path)
    paths_XYs = read_csv(processed_output_csv_path)
    plot(paths_XYs)
    regularized_paths = detect_and_regularize_shapes(paths_XYs)
    save_to_csv(regularized_paths, output_csv_path)

# Example usage:
csv_path = 'problems/isolated.csv'
processed_csv_path = 'isolated_sol.csv'
output_csv_path = 'data.csv'
main(csv_path, output_csv_path, processed_csv_path)