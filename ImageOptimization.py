import cv2
import numpy as np
import random


class Optimizer:
    def __init__(self, model):
        self.model = model

    # Regularization technique inspired from :
    # https://github.com/google/deepdream/blob/master/dream.ipynb
    # and
    # https://arxiv.org/pdf/1506.06579v1.pdf
    def make_step(self, image, objective_func, max_blur_iteration=10, l2_decay=0.01, pixel_clip=2,  **objective_params):
        jitter = 40
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        image = np.roll(np.roll(image, ox, -1), oy, -2)  # apply jitter shift
        #forward and backprop, objective function compute error gradient
        prediction = self.model.forward(image)
        prediction_grad = objective_func(prediction, **objective_params)
        image_grad = self.model.backward(prediction_grad)

        # for layer specific activation function

        """
        convos = self.model.get_convolution_activation()[4] # faces : 147, 192, 6
        convos[:, :, :, :] = 0
        convos[:, 147, :, :] = 1
        convos[:, 192, :, :] = 1
        convos[:, 6, :, :] = 1
        image_grad_1 = self.model.backward_layer(convos, 4)
        convos = self.model.get_convolution_activation()[3] # faces 117 245 42
        convos[:, :, :, :] = 0
        convos[:, 117, :, :] = 1
        convos[:, 245, :, :] = 1
        #convos[:, 42, :, :] = 1
        image_grad_2 = self.model.backward_layer(convos, 3)
        #optimize
        image_grad = image_grad_1 + image_grad_2
        """

        # Apply random gaussian blur on gradient
        rand_iter = random.randint(1, max_blur_iteration)
        rand_kernel = random.choice([3, 7, 9])
        image_grad = Optimizer.gaussian_blur(image_grad, iterations=rand_iter, kernel=rand_kernel)
        rand_step = random.uniform(0.7, 1.7)

        # Optimize
        grads = float(rand_step) / np.abs(image_grad).mean() * image_grad
        image += grads

        # Regularize image
        image = np.roll(np.roll(image, -ox, -1), -oy, -2)
        image = Optimizer.l2_decay(image, l2_decay)
        image = Optimizer.pixel_norm_clip(image, pixel_clip)

        return image

    @staticmethod
    def l2_decay(image, decay=0.01):
        image *= (1 - decay)
        return image

    @staticmethod
    def gaussian_blur(image, iterations=1, kernel=13):
        image = image[0, :, :, :].T
        for i in range(iterations):
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
        return image.T[np.newaxis, :, :, :]

    @staticmethod
    def pixel_norm_clip(image, threshold):
        ret = image.copy()
        norm = np.linalg.norm(ret, axis=(0, 1))
        ret[0, :, norm < threshold] = 0

        return ret

    @staticmethod
    def generate_gaussian_image(shape, RMean=127, GMean=127, BMean=127):
        image = np.zeros(shape, dtype=np.float32)
        image[:, :, 0] = RMean
        image[:, :, 1] = GMean
        image[:, :, 2] = BMean

        image += np.random.normal(0, 1, image.shape).astype(np.float32)
        return image

    # Objective functions
    @staticmethod
    def objective_L2(data):
        grad = data.copy()
        return grad

    @staticmethod
    def objective_maximize_class(data, index=0, energy=0.5):
        class_grad = data[0, index]
        grad = np.zeros(data.shape)
        grad[0, index] = (1. - class_grad) + 0.0000000001
        # if the image is otimized, invert the gradient
        if class_grad == 1:
            grad[0, index] = -energy
        return grad
