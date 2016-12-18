import cv2
import numpy as np


class Optimizer:
    def __init__(self, model):
        self.model = model

    def make_step(self, image, objective_func, i, step_size=1.5, **objective_params):
        jitter = 40
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        image = np.roll(np.roll(image, ox, -1), oy, -2)  # apply jitter shift
        #forward and backprop, objective function compute error gradient
        prediction = self.model.forward(image)
        prediction_grad = objective_func(prediction, **objective_params)
        image_grad = self.model.backward(prediction_grad)
        # for layer specific activation function
        #index = 4
        #image_grad = self.model.backward_layer(self.model.get_convolution_activation()[index], index)
        #optimize

        grads = float(step_size) / np.abs(image_grad).mean() * image_grad
        image += grads

        image = np.roll(np.roll(image, -ox, -1), -oy, -2)  # unshift image

        #regularize
        cv2.imshow("out", image[0, :, :, :].T)

        image = Optimizer.l2_decay(image, 0.01)
        cv2.imshow("l2", image[0, :, :, :].T)

        image = Optimizer.pixel_norm_clip(image, 1.5)
        cv2.imshow("clip", image[0, :, :, :].T)

        if i % 1 == 0:
            image = Optimizer.gaussian_blur(image, iterations=1, kernel=9)
            cv2.imshow("gaussian", image[0, :, :, :].T)

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
        print(class_grad)
        grad = np.zeros(data.shape)
        grad[:, :] = np.mean(grad)
        #grad *= energy
        grad[0, index] = class_grad * 10
        return grad
