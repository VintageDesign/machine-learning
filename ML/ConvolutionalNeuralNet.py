import numpy as np


class CNN:
    """
    CNN Network based on the blog post found here:
    https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
    """
    def __init__(self):
        self._bias = 0
        self._filter_1 = 0
        self._filter_2 = 0

    def _convolve(self, image, filter, stride=1):
        """
        Convolves the tensor of images with the tensor filter.
        :param image:
        :param stride:
        :return:
        """

        (rows_filt, cols_filt, depth_filter, _) = filter.shape
        (depth, rows_img, cols_img) = image.shape

        out_dim_rows = int((rows_img - depth_filter)/stride) + 1
        out_dim_cols = int((cols_img - depth_filter)/stride) + 1

        retval = np.zeros((rows_filt, out_dim, out_dim))

        for current_filt in range(rows_filt):
            current_y = 0
            out_y = 0

            while current_y + depth_filter <= rows_img:
                current_x = 0
                out_x = 0

                while current_x + depth_filter <= cols_img:
                    retval[current_filt, out_y, out_x] = np.sum(filter[current_filt] * image[:, current_y:current_y+depth_filter, current_x:current_x+depth_filter]) + bias[current_filt]
                    current_x += stride
                    out_x += 1
                current_y += stride
                out_y += stride

        return retval

    def _maxpool(self, layer, kernel_size=2, stride=1):
        """

        :param image:
        :param kernel_size:
        :param stride:
        :return:
        """

        (image_count, rows_image, cols_image) = layer.shape

        rows_pool = int((rows_image - kernel_size)/stride) + 1
        cols_pool = int((cols_image - kernel_size)/stride) + 1

        pool = np.zeros((image_count, rows_pool, cols_pool))

        for image in  range(image_count):
            current_y = 0
            out_y = 0

            while current_y + kernel_size <= rows_image:












