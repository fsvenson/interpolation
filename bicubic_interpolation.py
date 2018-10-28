#!/usr/bin/python3
"""Perform bicubic interpolation of a rectilinear image using
    a LUT describing the relationship of the rectilinear and barrel
    distorted coordinates."""
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate


def apply_barrel_distortion(x_coords, y_coords, width, height):
    """Apply barrel distortion on a given set of xy-coordinates."""
    x_center_origin = x_coords - width/2
    y_center_origin = y_coords - height/2
    x_squared = np.square(x_center_origin)
    y_squared = np.square(y_center_origin)
    radius = np.sqrt(x_squared + y_squared)
    phi = np.arctan2(y_center_origin, x_center_origin)

    k_1 = 6e-8
    k_2 = -2e-14
    radius_pow_2 = np.square(radius)
    radius_pow_4 = np.square(radius_pow_2)
    radius_distorted = np.multiply(
        radius, np.ones(radius.shape) - k_1 * radius_pow_2 + k_2 * radius_pow_4
    )
    x_distorted = np.multiply(radius_distorted, np.cos(phi)) + width/2
    y_distorted = np.multiply(radius_distorted, np.sin(phi)) + height/2

    return x_distorted, y_distorted


def fit_polynomials_to_coords(x_coords, y_coords):
    polynomials = np.zeros((3, y_coords.shape[0]))
    for i in range(0, polynomials.shape[1]):
        polynomials[:, i] = np.polyfit(x_coords, y_coords[i, :], 2)
    return polynomials


def estimate_mesh(rect_lut_x, rect_lut_y, dist_lut_x, dist_lut_y):
    x_polynomials = fit_polynomials_to_coords(rect_lut_y[:, 0], np.transpose(dist_lut_x))
    y_polynomials = fit_polynomials_to_coords(rect_lut_x[0, :], dist_lut_y)
    plt.figure(2)
    x_range = np.linspace(0, 2048, 5000)
    for i in range(0, y_polynomials.shape[1]):
        y = np.poly1d(y_polynomials[:, i])
        plt.plot(x_range, y(x_range), 'r')
    y_range = np.linspace(0, 800, 5000)
    for i in range(0, x_polynomials.shape[1]):
        x = np.poly1d(x_polynomials[:, i])
        plt.plot(x(y_range), y_range, 'r')
    plt.axis(aspect='image')
    plt.ylim(800, 0)
    plt.grid()

    return (x_polynomials, y_polynomials)


def sample_mesh(mesh, x, x_bin_size, y):
    x_poly = np.poly1d(mesh[0][:, x])
    y_poly = np.poly1d(mesh[1][:, y])

    y = y_poly(x*x_bin_size)
    x = x_poly(y)
    return np.array([x, y])


def grid_normalized_point(x, y, point, x_bin_size, y_bin_size):
    x_normalized = (point[0] - x * x_bin_size) / x_bin_size
    y_normalized = (point[1] - y * y_bin_size) / y_bin_size
    return (x_normalized, y_normalized)


def sample_interpolation_grid(mesh, point, x_bin_size, y_bin_size):
    x_grid_coord = point[0] / x_bin_size
    x_low = np.floor(x_grid_coord)
    x_high = np.ceil(x_grid_coord)

    if x_low == 0:
        x = np.array([0, 1, 2, 3])
    elif x_high == 32:
        x = np.array([29, 30, 31, 32])
    else:
        x = np.array([x_low - 1, x_low, x_high, x_high + 1])
    x = x.astype(int)

    y_grid_coord = point[1] / y_bin_size
    y_low = np.floor(y_grid_coord)
    y_high = np.ceil(y_grid_coord)

    if y_low == 0:
        y = np.array([0, 1, 2, 3])
    elif y_high == 8:
        y = np.array([5, 6, 7, 8])
    else:
        y = np.array([y_low - 1, y_low, y_high, y_high + 1])
    y = y.astype(int)

    sample_points = np.zeros((2, 16))
    for y_i in range(0, 4):
        for x_i in range(0, 4):
            sample_points[:, y_i*4+x_i] = sample_mesh(mesh, x[x_i], x_bin_size, y[y_i])

    return sample_points, grid_normalized_point(int(x_low), int(y_low), point, x_bin_size, y_bin_size)


def main():
    rectilinear_height = 800 + 1
    rectilinear_width = 2048 + 1
    rectilinear_x = np.arange(0, rectilinear_width)
    rectilinear_y = np.arange(0, rectilinear_height)
    rect_image_x, rect_image_y = np.meshgrid(rectilinear_x, rectilinear_y)
    x_bin_size = rectilinear_width/32
    y_bin_size = rectilinear_height/8
    rect_lut_x = rect_image_x[::int(y_bin_size),
                              ::int(x_bin_size)]
    rect_lut_y = rect_image_y[::int(y_bin_size),
                              ::int(x_bin_size)]

    dist_lut_x, dist_lut_y = apply_barrel_distortion(
        rect_lut_x, rect_lut_y, rectilinear_width - 1, rectilinear_height - 1)

    plt.figure(1)
    plt.subplot(311)
    plt.scatter(rect_lut_x, rect_lut_y)
    plt.scatter(dist_lut_x, dist_lut_y)

    plt.axis(aspect='image')
    plt.ylim(rectilinear_height, 0)
    plt.grid()

    # bicubic_interpolation_x = interpolate.RectBivariateSpline(
    #     rect_lut_y[:, 0], rect_lut_x[0, :], dist_lut_x, kx=3, ky=3)
    # bicubic_interpolation_y = interpolate.RectBivariateSpline(
    #     rect_lut_y[:, 0], rect_lut_x[0, :], dist_lut_y, kx=3, ky=3)
    bicubic_interpolation_x = interpolate.interp2d(
        rect_lut_x[0, :], rect_lut_y[:, 0], dist_lut_x, kind='cubic')
    bicubic_interpolation_y = interpolate.interp2d(
        rect_lut_x[0, :], rect_lut_y[:, 0], dist_lut_y, kind='cubic')

    dist_x_interp = bicubic_interpolation_x(rectilinear_x, rectilinear_y)
    dist_y_interp = bicubic_interpolation_y(rectilinear_x, rectilinear_y)
    dist_x, dist_y = apply_barrel_distortion(rect_image_x, rect_image_y,
                                             rectilinear_width - 1,
                                             rectilinear_height - 1)

    delta_x = dist_x_interp - dist_x
    delta_y = dist_y_interp - dist_y

    plt.subplot(312)
    plt.imshow(delta_x)
    plt.colorbar()
    plt.title('Residuals in X direction')
    plt.axis(aspect='image')
    plt.subplot(313)
    plt.imshow(delta_y)
    plt.colorbar()
    plt.title('Residuals in Y direction')
    plt.axis(aspect='image')

    mesh = estimate_mesh(rect_lut_x, rect_lut_y, dist_lut_x, dist_lut_y)

    sample_points = np.zeros((2, 9*33))
    for y in range(0, 9):
        for x in range(0, 33):
            sample_points[:, y*33+x] = sample_mesh(mesh, x, x_bin_size, y)

    plt.figure(2)
    plt.scatter(sample_points[0, :], sample_points[1, :], c='b', marker='x')

    interpolation_grid, normalized_point = sample_interpolation_grid(
        mesh, (1532, 257), x_bin_size, y_bin_size)
    plt.scatter(
        interpolation_grid[0, :], interpolation_grid[1, :], c='g', marker='o')
    print(normalized_point)

    plt.show()


if __name__ == '__main__':
    main()
