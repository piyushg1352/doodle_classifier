import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import svgwrite
from PIL import Image
from scipy.optimize import leastsq

# Load TensorFlow model
model_path = '/Users/ramjigupta/Desktop/adobemodel/doodle_classifier.keras'
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

data_cat = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

img_height = 180
img_width = 180

st.title('Doodle Image Classification and Shape Processing')

# Streamlit interface for image path input
image_path = st.text_input('Enter image path', '4532214299623424.png')

if st.button('Predict'):
    if os.path.exists(image_path):
        try:
            image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
            img_arr = tf.keras.utils.img_to_array(image)
            img_bat = tf.expand_dims(img_arr, 0)

            prediction = model.predict(img_bat)
            score = tf.nn.softmax(prediction)

            st.image(image)
            st.write(f'Doodle in image is {data_cat[np.argmax(score)]}.')
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error(f"File not found: {image_path}")
else:
    if not model:
        st.error("Model not loaded.")
    elif not os.path.exists(image_path):
        st.error(f"File not found: {image_path}")

# Define shape fitting functions
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
    num = b*b - a*c
    if num <= 0:
        return None
    cx = (c*d - b*f) / num
    cy = (a*f - b*d) / num
    up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b - a*c)*( ((c-a)*np.sqrt(1 + 4*b*b / ((a-c)*(a-c)))) - (c+a))
    down2 = (b*b - a*c)*( ((a-c)*np.sqrt(1 + 4*b*b / ((a-c)*(a-c)))) - (c+a))
    semi_major = np.sqrt(up / down1)
    semi_minor = np.sqrt(up / down2)
    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major
    theta = 0.5 * np.arctan(2 * b / (a - c))
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
            if isinstance(XY, np.ndarray) and XY.size > 0:
                W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'blue', 'green', 'yellow']

    for i, path in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in path:
            if isinstance(XY, np.ndarray) and XY.size > 0:
                path_data = [("M", (XY[0, 0], XY[0, 1]))]
                for j in range(1, len(XY)):
                    path_data.append(("L", (XY[j, 0], XY[j, 1])))
                if not np.allclose(XY[0], XY[-1]):
                    path_data.append(("Z", None))
                dwg_path = dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2)
                group.add(dwg_path)
            else:
                st.error(f"Unexpected type or empty data for path: {type(XY)}.")
    
    dwg.add(group)
    dwg.save()

    # Optionally convert SVG to PNG
    try:
        svg_image = Image.open(svg_path)
        png_path = svg_path.replace('.svg', '.png')
        fact = max(1, 1024 // min(H, W))
        width, height = fact * W, fact * H
        png_image = svg_image.resize((width, height), Image.ANTIALIAS)
        png_image.save(png_path, 'PNG')
        st.image(svg_image)
    except Exception as e:
        st.error(f"Error converting SVG to PNG: {e}")

def main(input_csv_path, output_csv_path, processed_output_csv_path):
    process_data(input_csv_path, processed_output_csv_path)
    paths_XYs = read_csv(processed_output_csv_path)
    plot(paths_XYs)
    regularized_paths = detect_and_regularize_shapes(paths_XYs)
    save_to_csv(regularized_paths, output_csv_path)
    polylines2svg(regularized_paths, 'svg/shapes.svg')

# Streamlit interface for CSV paths
path_csv = st.text_input('Path to input CSV file : ', 'problems/isolated.csv')
st.write('\n')
split = st.text_input('Enter name for new csv file : ','example')
csv_path = path_csv
csv_save = st.text(f'Path to save new CSV file : problem_solution/{split}.csv')
path_csv_save=f'problem_solution/{split}.csv'
processed_csv_path = path_csv_save
output_csv_path = st.text_input('Path to output CSV file', 'problem_solution/newData.csv')

if st.button('Run Processing'):
    if os.path.exists(csv_path):
        main(csv_path, output_csv_path, processed_csv_path)
        st.success('Processing complete!')
    else:
        st.error(f"Input CSV file not found: {csv_path}")