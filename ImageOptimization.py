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
    def make_step(self, image, objective_func, max_blur_iteration=10, l2_decay=0.01, pixel_clip=2,
                  sorted_activation=None,
                  **objective_params):
        jitter = 40
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        image = np.roll(np.roll(image, ox, -1), oy, -2)  # apply jitter shift
        #forward and backprop, objective function compute error gradient
        prediction = self.model.forward(image)
        prediction_grad = objective_func(prediction, **objective_params)
        convos, linears = self.model.get_activation()
        image_grad = np.zeros(image.shape).astype(np.float64)
        # Do normal backprop
        if sorted_activation is None:
            image_grad = self.model.backward(prediction_grad)
            norm_im = image_grad[0, :, :, :].T.copy()
            cv2.normalize(image_grad[0, :, :, :].T, norm_im, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Do backprop based on most activated units
        else:
            convo_indexes, linear_indexes = sorted_activation
            for i, activation in enumerate(convo_indexes):
                if i > 3:
                    convos_grad = np.zeros(convos[i].shape, dtype=convos[i].dtype)
                    convos_grad[:, activation[:7], :, :] = 0.3 * i
                    grad = self.model.backward_layer(convos_grad, i)
                    image_grad += grad
            for i, activation in enumerate(linear_indexes):
                if i <= 0:
                    linear_grad = np.zeros(linears[i].shape, dtype=linears[i].dtype)
                    linear_grad[:, activation[:5]] = linears[i][:, activation[:5]]
                    grad = self.model.backward_layer(linear_grad, i)
                    image_grad += grad


        # Apply random gaussian blur on gradient
        rand_iter = random.randint(3, max_blur_iteration)
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
