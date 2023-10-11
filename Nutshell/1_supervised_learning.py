import os
import cv2
import yaml
import numpy
import random
import argparse
from tqdm import tqdm
import tensorflow.keras

import classify_noise
import image_noise_addition


# Function to extract a path using the config file
def extract_path(config, noise_type, model_path=False):
    output_path = os.path.join(config['sl']['storage_dir'], noise_type, config['data']['colormode'])
    if model_path:
        return os.path.join(output_path, "models", config['sl']['model'])
    else:
        return output_path


# Function to read the images from the path specified in the config file (dataset can be either train or test)
def read_images(config, dataset):
    image_size = retrieve_image_size(config)
    path = os.path.join(config['data']['original_dir'], dataset)
    filepaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]

    if config['data']['colormode'] == "grayscale":
        images = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE), image_size, interpolation = cv2.INTER_AREA) for f in filepaths]
    elif config['data']['colormode'] == "rgb":
        images = [cv2.resize(cv2.imread(f, flags=cv2.IMREAD_COLOR), image_size, interpolation = cv2.INTER_AREA) for f in filepaths]
    return images


# Function to get the user specified image size
def retrieve_image_size(config):
    return (config['data']['shape']['height'], config['data']['shape']['width'])


# Function to get the user specified variations of noise levels
def retrieve_variances(config, noise_type):
    noise_type_settings = config['noise']['settings'][noise_type]
    variances = numpy.linspace(0.0, noise_type_settings['max_variance'], noise_type_settings['num_of_variances'])
    variances = [round(variance, 1) for variance in variances]
    return variances


# Functions to apply a type of noise to an image
def apply_blur(image, level):
    return image_noise_addition.Noise.blur(image, level)


def apply_gaussian_noise(image, level, config):
    if config['data']['colormode'] == 'grayscale':
        return image_noise_addition.Noise.gauss(image, 0, level)
    else:
        return image_noise_addition.Noise.gauss_multicolor(image, 0, level)


# Function to read the training images, apply multiple levels of noise to each one and store them in a directory structure that is compatible with keras
def generate(config, noise_type):
    images = read_images(config, "train")
    variances = retrieve_variances(config, noise_type)
    output_path = extract_path(config, noise_type, model_path=False)

    for variance in variances:
        variance_dir = os.path.join(output_path, 'data', 'variance' + str(variance))
        os.makedirs(variance_dir, exist_ok=True)
        for i in tqdm(range(len(images)), desc=f"Variance {variance}"):
            # A minimal amount of the secondary type of noise is applied to create more robust classifiers
            if noise_type == 'gaussian':
                blur_level = config['noise']['settings']['blur']['max_variance']
                blur_level = random.uniform(0.0, float(blur_level/4))
                image = apply_blur(images[i], blur_level)
                image = apply_gaussian_noise(image, variance, config)
            elif noise_type == 'blur':
                gaussian_level = config['noise']['settings']['gaussian']['max_variance']
                gaussian_level = random.uniform(0.0, float(gaussian_level/4))
                image = apply_blur(images[i], variance)
                image = apply_gaussian_noise(image, gaussian_level, config)
           
            filename = 'image_' + str(i) + '_variance_' + str(variance) + '.jpg'
            image_path = os.path.join(variance_dir, filename)
            cv2.imwrite(image_path, image)


# Function to train a model that estimates the applied noise level
def train(config, noise_type):
    image_size = retrieve_image_size(config)
    data_path = os.path.join(extract_path(config, noise_type, model_path=False), 'data')
    # generate the data transformation for keras (tensor)
    train_ds, val_ds = classify_noise.NNUtilities.generate_keras_ds(data_path, [f for f in os.listdir(data_path)], image_size, config['data']['colormode'])
    # prepare memory allocation for faster processing
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    variances = retrieve_variances(config, noise_type)
    # instantiate model class own library (additional models can be inserted below)
    if config['sl']['model'] == 'lenet':
        model = classify_noise.NNModel.make_model_lenet5_original(image_size, config['data']['colormode'], len(variances), 2)
    elif config['sl']['model'] == 'vgg16':
        model = classify_noise.NNModel.make_model_vgg16_reg(image_size, config['data']['colormode'], len(variances), 2)
    
    model_dir = extract_path(config, noise_type, model_path=True)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "training_log.txt")
    model_path = os.path.join(model_dir, "model.h5")

    # Log all of the training results, but store only the model that generalizes better
    callbacks = [
        tensorflow.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tensorflow.keras.callbacks.CSVLogger(filename=log_path)
    ]
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=config['sl']['epochs'], callbacks=callbacks, validation_data=val_ds,
    )


# Function to evaluate the performance of a model on the previously unseen test data
def evaluate(config, noise_type):
    images = read_images(config, "test")
    variances = retrieve_variances(config, noise_type)
    model_dir = extract_path(config, noise_type, model_path=True)
    log_path = os.path.join(model_dir, "evaluation_log.txt")
    fp = open(log_path, "w")

    model_path = os.path.join(model_dir, "model.h5")
    noise_classifier = tensorflow.keras.models.load_model(model_path)

    for variance in variances:
        estimates = []
        correct_predictions = 0
        correct_approximate_predictions = 0
        for iteration in tqdm(range(config['sl']['evaluate_iterations']), desc=f"Variance {variance}"):
            summed_gaussian_certainty = 0
            image_index = random.randint(0, len(images)-1)
            for iteration2 in range(config['sl']['evaluate_average_count']):
                if noise_type == 'gaussian':
                    blur_level = config['noise']['settings']['blur']['max_variance']
                    blur_level = random.uniform(0.0, float(blur_level/4))
                    image = apply_blur(images[image_index], blur_level)
                    image = apply_gaussian_noise(image, variance, config)
                elif noise_type == 'blur':
                    gaussian_level = config['noise']['settings']['gaussian']['max_variance']
                    gaussian_level = random.uniform(0.0, float(gaussian_level/4))
                    image = apply_blur(images[image_index], variance)
                    image = apply_gaussian_noise(image, gaussian_level, config)

                # Apply classifier
                img_array = tensorflow.keras.preprocessing.image.img_to_array(image)
                img_array = tensorflow.expand_dims(img_array, 0)
                prediction = noise_classifier.predict(img_array, verbose=0)[0]
                # Check if prediction is precise
                if numpy.argmax(prediction) * variances[1] == variance:
                    correct_predictions += 1
                # Check if prediction is approximate
                if abs(numpy.argmax(prediction) * variances[1] - variance) <= variances[1] * 2:
                    correct_approximate_predictions += 1

                # Interpret classifier
                gaussian_certainty = 0
                for index in range(len(variances)):
                    gaussian_certainty += variances[index] * prediction[index]
                summed_gaussian_certainty += gaussian_certainty
            estimates.append(summed_gaussian_certainty / config['sl']['evaluate_average_count'])

        fp.write("Image with noise level {0:2.2f}: classified in range {1: 1.2f} .. {2: 1.2f}; range width {3: 1.2f}\n".format(variance, min(estimates), max(estimates), max(estimates) - min(estimates)))
        fp.write(f"Number of correct predictions: {correct_predictions} ({float(correct_predictions/config['sl']['evaluate_iterations'])})\n")
        fp.write(f"Number of correct approximate predictions: {correct_approximate_predictions} ({float(correct_approximate_predictions/config['sl']['evaluate_iterations'])})\n\n")
    fp.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--generate', action='store_true', help='1. Generate images')
    arg_parser.add_argument('--train',    action='store_true', help='2. Train the classifier')
    arg_parser.add_argument('--evaluate', action='store_true', help='3. Evaluate the classifier')
    arg_parser.add_argument('--all', action='store_true', help='4. Perform all of the above steps')

    arg_parser.add_argument('--gaussian', action='store_true', help='Specify type of noise [gaussian/blur]')
    arg_parser.add_argument('--blur', action='store_true', help='Specify type of noise [gaussian/blur]')

    args = arg_parser.parse_args()
    if True not in [args.generate, args.train, args.evaluate, args.all]:
        arg_parser.print_help()

    if True not in [args.gaussian, args.blur]:
        arg_parser.print_help()
        quit()
    else:
        noise_type = 'gaussian' if args.gaussian else 'blur'

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)   

    if args.generate:
        generate(config, noise_type)
    if args.train:
        train(config, noise_type)
    if args.evaluate:
        evaluate(config, noise_type)
    if args.all:
        generate(config, noise_type)
        train(config, noise_type)
        evaluate(config, noise_type)
