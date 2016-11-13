import math
import numpy as np
import cv2


class SpatialActivationViewer:
    def __init__(self):
        self.layer_selected = 0
        self.n_layers = 0
        self.filter_selected = None

    def filter_selection(self, x, y):
        """
        Convert screen coordinate to grid coordinate and save state as a user selected grid cell
        :param x:
        :param y:
        :return:
        """
        w, h = self.filter_size[self.layer_selected]
        filter_x, filter_y = self.pixel_to_grid(x, y, h, w)
        self.filter_selected = (filter_x, filter_y)

    def pixel_to_grid(self, x, y, height, width):
        grid_x = float(x - 1) / float(width + 2)
        grid_y = float(y - 1) / float(height + 2)
        return int(grid_x), int(grid_y)

    def grid_to_pixel(self, grid_x, grid_y, height, width):
        """
        Compute the screen position given a grid emplacement. It calculates a contour of 1 pixel around each cell
        :param mosaic_x:
        :param mosaic_y:
        :return:
        """
        x = int(grid_x) * (2 + width) + 1
        y = int(grid_y) * (2 + height) + 1
        return x, y

    def layer_selection_increment(self, inc):
        self.layer_selected = (self.layer_selected + inc) % self.n_layers

    def update_filter_data(self, layers_filters):
        self.n_layers = len(layers_filters)
        self.n_filter_per_layer = []
        self.mosaic_sizes = []
        self.border_sizes = []
        self.filter_size = []
        self.filter_grid = []
        for filter in layers_filters:
            self.n_filter_per_layer.append(len(filter))
            filter = np.squeeze(filter)
            n, w, h = filter.shape
            mosaic_s = math.ceil(math.sqrt(n))
            self.mosaic_sizes.append(mosaic_s)
            self.border_sizes.append(2 * mosaic_s)
            self.filter_size.append((w, h))
            filter_grid = np.zeros((mosaic_s * w + 2 * mosaic_s,
                               mosaic_s * h + 2 * mosaic_s, 3),
                              dtype=np.uint8)
            filter_grid[:, :, 0].fill(100)
            self.filter_grid.append(filter_grid)

    def draw(self, filters_layer):
        filters = filters_layer[self.layer_selected]
        grid_size = self.mosaic_sizes[self.layer_selected]
        w, h = self.filter_size[self.layer_selected]
        filters = np.squeeze(filters)
        filter_img = None
        for i, individual_filter in enumerate(filters):
            mosaic_x = (i % grid_size)
            mosaic_y = (i / grid_size)
            x, y = self.grid_to_pixel(mosaic_x, mosaic_y, h, w)
            self.filter_grid[self.layer_selected][y:y + h, x:x + w, :] = np.repeat(np.expand_dims(cv2.convertScaleAbs(individual_filter), axis=2), 3, axis=2)
            if self.filter_selected:
                if int(mosaic_x) == self.filter_selected[0] and int(mosaic_y) == self.filter_selected[1]:
                    filter_img = np.repeat(np.expand_dims(cv2.convertScaleAbs(individual_filter), axis=2), 3,
                                                        axis=2).copy()
        cpy = self.filter_grid[self.layer_selected].copy()
        if self.filter_selected:
            select_x, select_y = self.filter_selected
            x, y = self.grid_to_pixel(select_x, select_y, h, w)
            cv2.rectangle(cpy,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0))
        return cpy, filter_img

    def get_layer_mean_activation(self, filters_layer):
        filters = filters_layer[self.layer_selected]
        mean = 0
        for filter in filters:
            mean += np.mean(filter)
        return mean/len(filters)

    def get_selected_filter_mean_activation(self, filters_layer):
        if self.filter_selected is None:
            raise Exception("No filter selected")
        select_x, select_y = self.filter_selected
        index = select_y * self.mosaic_sizes[self.layer_selected] + select_x
        return np.mean(filters_layer[self.layer_selected][:, index, :, :])