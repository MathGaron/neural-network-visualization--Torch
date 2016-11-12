import math
import numpy as np
import cv2


class SpatialActivationViewer:
    def __init__(self):
        self.layer_selected = 0
        self.n_layers = 0
        self.filter_selected = None

    def filter_selection(self, x, y):
        filter_x = int(math.ceil(float(x-1)/float(self.heights[self.layer_selected] + 2)))
        filter_y = int(math.ceil(float(y-1)/float(self.widths[self.layer_selected] + 2)))
        print x, y
        print filter_x, filter_y
        self.filter_selected = (filter_x, filter_y)

    def layer_selection_increment(self, inc):
        self.layer_selected = (self.layer_selected + inc) % self.n_layers

    def update_filter_data(self, layers_filters):
        self.n_layers = len(layers_filters)
        self.n_filter_per_layer = []
        self.mosaic_sizes = []
        self.border_sizes = []
        self.heights = []
        self.widths = []
        self.mosaics = []
        for filter in layers_filters:
            self.n_filter_per_layer.append(len(filter))
            filter = np.squeeze(filter)
            n, w, h = filter.shape
            mosaic_s = math.ceil(math.sqrt(n))
            self.mosaic_sizes.append(mosaic_s)
            self.border_sizes.append(2 * mosaic_s)
            self.heights.append(h)
            self.widths.append(w)
            mosaic = np.zeros((mosaic_s * w + 2 * mosaic_s,
                               mosaic_s * h + 2 * mosaic_s, 3),
                              dtype=np.uint8)
            mosaic[:, :, 0].fill(255)
            self.mosaics.append(mosaic)

    def mosaic_to_pixel(self, mosaic_x, mosaic_y):
        border_x = 1 + 2 * mosaic_x
        border_y = 1 + 2 * mosaic_y
        x = (int(mosaic_x) * self.heights[self.layer_selected]) + border_x
        y = (int(mosaic_y) * self.widths[self.layer_selected]) + border_y
        return x, y

    def draw(self, filters_layer):
        filters = filters_layer[self.layer_selected]
        filters = np.squeeze(filters)
        filter_img = None
        for i, individual_filter in enumerate(filters):
            mosaic_x = (i % self.mosaic_sizes[self.layer_selected])
            mosaic_y = (i / self.mosaic_sizes[self.layer_selected])
            x, y = self.mosaic_to_pixel(mosaic_x, mosaic_y)
            self.mosaics[self.layer_selected][x:x + self.heights[self.layer_selected], y:y + self.widths[self.layer_selected], :] = np.repeat(np.expand_dims(cv2.convertScaleAbs(individual_filter), axis=2), 3,
                                                    axis=2)
            if self.filter_selected:
                if int(mosaic_x)== self.filter_selected[0] and int(mosaic_y) == self.filter_selected[1]:
                    filter_img = np.repeat(np.expand_dims(cv2.convertScaleAbs(individual_filter), axis=2), 3,
                                                        axis=2).copy()
        cpy = self.mosaics[self.layer_selected].copy()
        if self.filter_selected:
            select_x, select_y = self.filter_selected
            x, y = self.mosaic_to_pixel(select_x, select_y)
            cv2.rectangle(cpy,
                          (x, y),
                          (x + self.widths[self.layer_selected], y + self.heights[self.layer_selected]),
                          (0, 255, 0))
        return cpy, filter_img