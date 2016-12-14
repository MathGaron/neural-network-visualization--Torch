import cv2
import numpy as np


class Optimizer:
    def __init__(self, model):
        self.model = model

    def make_step(self, image, objective_func, step_size=1.5, **objective_params):
        #forward and backprop, objective function compute error gradient
        prediction = self.model.forward(image)
        prediction_grad = objective_func(prediction, **objective_params)
        image_grad = self.model.backward(prediction_grad)
        # for layer specific activation function
        #index = 4
        #image_grad = self.model.backward_layer(self.model.get_convolution_activation()[index], index)
        #optimize
        image += float(step_size) / np.abs(image_grad).mean() * image_grad
        #regularize
        image = Optimizer.l2_decay(image, 0.01)
        image = Optimizer.gaussian_blur(image)
        image = Optimizer.pixel_norm_clip(image, 9)
        return image

    @staticmethod
    def l2_decay(image, decay=0.01):
        image *= (1 - decay)
        return image

    @staticmethod
    def gaussian_blur(image):
        return cv2.GaussianBlur(image[0, :, :, :].T, (3, 3), 0).T

    @staticmethod
    def pixel_norm_clip(image, threshold):
        norm = np.linalg.norm(image, axis=(0, 1))
        image[0, :, norm < threshold] = 0
        return image

    @staticmethod
    def generate_gaussian_image(shape):
        image = np.zeros(shape, dtype=np.uint8)
        image[:, :, :] = 255 / 2
        image += np.random.normal(0, 5, image.shape).astype(np.uint8)
        return image

    # Objective functions
    @staticmethod
    def objective_L2(data):
        grad = data.copy()
        return grad

    @staticmethod
    def objective_maximize_class(data, index=0, energy=0.5):
        grad = np.zeros(data.shape)
        grad[0, index] = energy
        return grad
