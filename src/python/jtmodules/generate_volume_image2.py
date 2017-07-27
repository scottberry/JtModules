# Copyright 2017 Scott Berry, University of Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Jterator module for extracting a volume image from a 3D stack'''
import collections
import logging
import numpy as np
import mahotas as mh
from jtlib.filter import log_2d, log_3d
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

VERSION = '0.2.0'

Output = collections.namedtuple('Output', ['volume_image', 'figure'])


def array_to_coordinate_list(array):
    '''Convert a 2D array representation of points in 3D
    to a list of x,y,z coordinates'''
    points = []
    for ix in range(array.shape[0]):
        for iy in range(array.shape[1]):
            if (array[ix, iy] > 0):
                points.append((ix, iy, array[ix, iy]))
    return points


def subsample_coordinate_list(points, num):
    subpoints = np.array(points)[np.linspace(
        start=0, stop=len(points), endpoint=False,
        num=num, dtype=np.uint32)]
    return list(map(tuple, subpoints))


def plane(x, y, params):
    '''Compute z-coordinate of plane in 3D'''
    a, b, c = params
    z = (a * x) + (b * y) + c
    return z


# Least squares error estimate
def squared_error(params, points):
    '''Compute the sum of squared residuals'''
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


def fit_plane(points):
    '''Fit a plane to the 3D beads surface'''
    import scipy.optimize
    import functools

    fun = functools.partial(squared_error, points=points)
    params0 = [0, 0, 0]
    return scipy.optimize.minimize(fun, params0)


def interpolate_surface(coords, output_shape, method='linear'):
    '''Given a set of coordinates (not necessarily on a grid), an
    interpolation is returned as a numpy array'''
    from scipy.interpolate import griddata

    xy = np.column_stack((coords[:, 0], coords[:, 1]))
    xv, yv = np.meshgrid(
        range(output_shape[0]),
        range(output_shape[1])
    )
    if method == 'nearest':
        interpolate = griddata(
            xy, np.array(coords[:, 2]), (xv, yv), method='nearest', rescale=False
        )
    elif method == 'cubic':
        interpolate = griddata(
            xy, np.array(coords[:, 2]), (xv, yv), method='cubic', rescale=False
        )
    elif method == 'linear':
        interpolate = griddata(
            xy, np.array(coords[:, 2]), (xv, yv), method='linear', rescale=False
        )

    return interpolate.T


def main(image, mask, threshold=150, bead_size=3,
         filter_type='log_2d', bead_localisation='max',
         minimum_bead_intensity=150,
         outlier_detection=True, outlier_tolerance=4,
         close_surface=False, close_disc_size=8, plot=False):
    '''Converts an image stack with labelled cell surface to a cell
    `volume` image

    Parameters
    ----------
    image: numpy.ndarray[Union[numpy.uint8, numpy.uint16]]
        grayscale image in which beads should be detected (3D)
    mask: numpy.ndarray[Union[numpy.int32, numpy.bool]]
        binary or labeled image of cell segmentation (2D)
    threshold: int, optional
        intensity of bead (default: ``150``)
    bead_size: int, optional
        minimal size of bead (default: ``3``)
    filter_type: str, optional
        filter used to emphasise the beads in 3D
        (options: ``log_2d`` (default) or ``log_3d``)
    bead_localisation: str, optional
        method used to localise the beads position within the filtered
        regions (options: ``max`` (default) or ``centroid``)
    minimum_bead_intensity: int, optional
        minimum intensity in the original image of an identified bead
        centre. Use to filter low intensity beads.
    outlier_detection: bool, optional
        replace outliers with aberrant z-steps
    outlier_tolerance: int, optional
        maximum number of z-steps between neighboring beads
    close_surface: bool, optional
        whether the interpolated surface should be morphologically closed
    close_disc_size: int, optional
        size in pixels of the disc used to morphologically close the
        interpolated surface
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    jtmodules.generate_volume_image.Output
    '''

    n_slices = image.shape[-1]
    logger.debug('input image has z-dimension %d', n_slices)

    # Remove high intensity pixels
    detect_image = image.copy()
    p = np.percentile(detect_image, 99.9)
    detect_image[detect_image > p] = p

    # Perform LoG filtering in 3D to emphasise beads
    if filter_type == 'log_2d':
        logger.info('using stacked 2D LoG filter to detect beads')
        f = -1 * log_2d(size=bead_size, sigma=float(bead_size - 1) / 3)
        filt = np.stack([f for _ in range(2 * bead_size)], axis=2)

    elif filter_type == 'log_3d':
        logger.info('using 3D LoG filter to detect beads')
        filt = -1 * log_3d(bead_size, (float(bead_size - 1) / 3,
                                       float(bead_size - 1) / 3,
                                       4 * float(bead_size - 1) / 3))
    else:
        logger.info('using unfiltered image to detect beads')

    if filter_type == 'log_2d' or filter_type == 'log_3d':
        logger.debug('convolve image with filter kernel')
        detect_image = mh.convolve(detect_image.astype(float), filt)
        detect_image[detect_image < 0] = 0

    logger.debug('label and filter beads by size')
    labeled_beads, n_labels = mh.label(detect_image > threshold)
    logger.info('detected %d beads', n_labels)

    sizes = mh.labeled.labeled_size(labeled_beads)
    too_small = np.where(sizes < 2 * bead_size)
    labeled_beads = mh.labeled.remove_regions(labeled_beads, too_small)
    mh.labeled.relabel(labeled_beads, inplace=True)
    logger.info('%d beads remain after filtering', np.max(labeled_beads))

    if bead_localisation == 'centroids':
        logger.debug('localise beads in 3D by centroid')
        bead_coords = mh.center_of_mass(labeled_beads, labels=labeled_beads)
        bead_coords = bead_coords[1:, :].astype(int)
    elif bead_localisation == 'max':
        logger.debug('localise beads in 3D by maximum intensity')
        bead_coords = []
        bboxes = mh.labeled.bbox(labeled_beads)
        for bead in range(np.max(labeled_beads)):
            x_min, x_max, y_min, y_max, z_min, z_max = bboxes[bead]
            label_image = labeled_beads[x_min:x_max, y_min:y_max, z_min:z_max]
            bounded = image[x_min:x_max, y_min:y_max, z_min:z_max]
            bounded = np.copy(bounded)
            bounded[label_image != bead] = 0
            local_coords = np.unravel_index(
                bounded.argmax(),
                bounded.shape
            )
            centre = (np.array([x_min, y_min, z_min]) +
                      np.asarray(local_coords, dtype=np.int32))
            if (image[centre[0], centre[1], centre[2]] >
                    minimum_bead_intensity):
                bead_coords.append(centre)
        bead_coords = np.asarray(bead_coords)
    else:
        logger.error('unidentified bead localisation method')

    if outlier_detection:
        logger.debug('update outliers using local distance detection' +
                     ' and median replacement')
        # use KDTree to efficiently get the nearest neighbours
        neighbours_xy = KDTree(bead_coords[:, 0:2])
        for bead in range(bead_coords.shape[0]):
            q = neighbours_xy.query(bead_coords[bead, 0:2], k=4, p=2)
            k0, k1, k2, k3 = bead_coords[q[1]]
            max_z = max(
                abs(k0[2] - k1[2]),
                abs(k0[2] - k2[2]),
                abs(k0[2] - k3[2])
            )
            max_other = max(
                abs(k1[2] - k2[2]),
                abs(k2[2] - k3[2]),
                abs(k3[2] - k1[2])
            )
            if max_z > max_other + outlier_tolerance:
                bead_coords[bead, 2] = np.median([k1[2], k2[2], k3[2]])

    logger.debug('convert %d bead vertices to image', bead_coords.shape[0])
    coord_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
    for i in range(bead_coords.shape[0]):
        coord_image[bead_coords[i][0], bead_coords[i][1]] = bead_coords[i][2]

    logger.debug('mask beads inside cells')
    slide = np.copy(coord_image)
    slide[mask > 0] = 0

    logger.debug('determine coordinates of slide surface')
    slide_coordinates = array_to_coordinate_list(slide)
    bottom_surface = fit_plane(subsample_coordinate_list(
        slide_coordinates, 2000)
    )

    logger.debug('subtract slide surface to get absolute bead coordinates')
    bead_coords_abs = bead_coords.copy()
    for i in range(bead_coords_abs.shape[0]):
        bead_height = (
            int(bead_coords_abs[i, 2] -
                plane(
                    bead_coords_abs[i, 0],
                    bead_coords_abs[i, 1],
                    bottom_surface.x
            ))
        )
        bead_coords_abs[i, 2] = bead_height if bead_height > 0 else 0

    logger.info('interpolate cell surface')
    volume_image = interpolate_surface(
        coords=bead_coords_abs,
        output_shape=np.shape(image[:, :, 1]),
        method='linear'
    )

    volume_image = volume_image.astype(image.dtype)

    if (close_surface is True):
        logger.info('morphological close cell surface')
        volume_image = mh.close(
            volume_image,
            Bc=mh.disk(close_disc_size)
        )

    logger.debug('set regions outside mask to zero')
    volume_image[mask == 0] = 0

    if plot:
        logger.debug('convert bottom surface plane to image for plotting')
        bottom_surface_image = np.zeros(slide.shape, dtype=np.uint8)
        for ix in range(slide.shape[0]):
            for iy in range(slide.shape[1]):
                bottom_surface_image[ix, iy] = plane(
                    ix, iy, bottom_surface.x)
        outlines = mh.labeled.bwperim(np.max(labeled_beads, axis=2) > 0)
        logger.info('create plot')
        from jtlib import plotting
        plots = [
            plotting.create_intensity_overlay_image_plot(
                np.max(image, axis=2), outlines, 'ul', clip=True
            ),
            plotting.create_intensity_image_plot(
                bottom_surface_image, 'll', clip=True
            ),
            plotting.create_intensity_image_plot(
                volume_image, 'ur', clip=True
            )

        ]
        figure = plotting.create_figure(
            plots, title='Convert stack to volume image'
        )
    else:
        figure = str()

    return Output(volume_image, figure)
