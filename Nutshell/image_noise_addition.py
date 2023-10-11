import cv2
import numpy
import PIL
import math


class Noise:
    @staticmethod
    def gauss(image, mean, var):
        row, col = image.shape
        sigma = var**0.5
        gauss = numpy.random.normal(mean, sigma, (row, col)).astype('uint8')
        gauss = gauss.reshape(row, col)
        noisy = cv2.add(image, gauss)
        return noisy
    
    @staticmethod
    def gauss_multicolor(image, mean, var):
        row, col, ch= image.shape
        sigma = var**0.5
        gauss = numpy.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
        gauss = gauss.reshape(row, col, ch)
        noisy = cv2.add(image, gauss)
        return noisy
    
    @staticmethod
    def distortion(image, variance, angle, max_var):
        rows, cols = image.shape
        pad_len = int(rows/2)
        # Add black border to avoid gaps in corners after wave tranformation
        padded_img = numpy.pad(image, pad_len, 'constant', constant_values=0)
        padded_img = PIL.Image.fromarray(padded_img)
        padded_img = padded_img.rotate(angle)
        padded_img = numpy.array(padded_img)

        # Add aberration
        aberrated_img = numpy.zeros(padded_img.shape, dtype=image.dtype)
        padded_rows, padded_cols = padded_img.shape
        for i in range(padded_rows):
            for j in range(padded_cols):
                offset_x = int(25.0 * math.sin(variance * 3.14 * i / max_var))
                aberrated_img[i,j] = padded_img[i,(j+offset_x)%padded_cols]

        # Unrotate aberrated image and remove border
        aberrated_img = PIL.Image.fromarray(aberrated_img)
        aberrated_img = aberrated_img.rotate(-angle)
        aberrated_img = numpy.array(aberrated_img)
        aberrated_img = aberrated_img[pad_len:-pad_len,pad_len:-pad_len]
        return numpy.array(aberrated_img)

    # (from https://stackoverflow.com/questions/59991178/creating-pixel-noise-with-pil-python)
    @staticmethod
    def add_salt_and_pepper(image, amount):
        output = numpy.copy(numpy.array(image))

        # add salt
        nb_salt = numpy.ceil(amount * output.size * 0.5)
        coords = [numpy.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
        output[tuple(coords)] = 1

        # add pepper
        nb_pepper = numpy.ceil(amount * output.size * 0.5)
        coords = [numpy.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
        output[tuple(coords)] = 0

        return PIL.Image.fromarray(output)

    @staticmethod
    def blur(matrix, blur_level):
        image = PIL.Image.fromarray(matrix)
        transformed = image.filter(PIL.ImageFilter.GaussianBlur(blur_level))
        return numpy.array(transformed)

    @staticmethod
    def change_sharpness(image, sharpness_level):
        enhancer = PIL.ImageEnhance.Sharpness(PIL.Image.fromarray(image))
        # sharpness_level = 2 improves the 'sharpness'
        transformed = enhancer.enhance(sharpness_level)
        return numpy.array(transformed)

    @staticmethod
    def affine(matrix, skew_level):
        # level  = 0.3 is nicely skewed ('top of the image to the right')
        image = PIL.Image.fromarray(matrix)
        transformed = image.transform((image.width, image.height), PIL.Image.AFFINE, (1, skew_level, 0, 0, 1, 0))
        # Data is a 6-tuple (a, b, c, d, e, f) which contain the first two rows from an affine transform matrix.
        # Each output pixel (x, y) comes from input pixel (a x + b y + c, d x + e y + f)
        return numpy.array(transformed)

    @staticmethod
    def rotate(image, angle):
        return image.rotate(angle)
        # Rotate given image matrix
        # img = PIL.Image.fromarray(matrix)
        # transformed = img.rotate(angle)
        # return numpy.array(transformed)

    @staticmethod
    def crop(image, x_size, y_size):
        return image.crop(((image.width - x_size) / 2, (image.height - y_size) / 2, (image.width + x_size) / 2, (image.height + y_size) / 2))

    @staticmethod
    # this belongs in Utilities
    def resize(image, x_pixels, y_pixels):
        return image.resize((x_pixels, y_pixels), PIL.Image.ANTIALIAS)

    @staticmethod
    def change_brightness(image, brightness_level):
        # brightness_level = 1.0 gives original image
        enhancer = PIL.ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_level)

    @staticmethod
    def change_contrast(image, contrast_level):
        # contrast_level = 0.5 decreases contrast
        enhancer = PIL.ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_level)
