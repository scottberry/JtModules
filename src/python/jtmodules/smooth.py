import logging
import cv2
import mahotas as mh
import skimage.morphology
import skimage.filters.rank
import numpy as np

VERSION = '0.0.1'

logger = logging.getLogger(__name__)


def main(image, filter_name, filter_size, sigma=0, sigma_color=0,
                 sigma_space=0, plot=False):
    '''Smoothes (blurs) an image using a low-pass filter.

    For more information on image filtering see
    `OpenCV tutorial <http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_smoothed_imageproc/py_filtering/py_filtering.html>`_.

    Parameters
    ----------
    image: numpy.ndarray
        grayscale image that should be smoothed
    filter_name: str
        name of the filter kernel that should be applied
        (options: ``{"avarage", "gaussian", "median", "median-bilateral", "gaussian-bilateral"}``)
    filter_size: int
        size (width/height) of the kernel (must be an odd, positive integer)
    sigma_color: int, optional
        Gaussian component (sigma) applied in the intensity domain
        (color space) - only relevant for "bilateral" filter (default: ``0``)
    sigma_space: int, optional
        Gaussian component (sigma) applied in the spacial domain
        (coordinate space) - only relevant for "bilateral" filter
        (default: ``0``)
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    Dict[str, numpy.ndarray[numpy.int32] or str]
        * "smoothed_image": smoothed intensity image
        * "figure": html string in case `plot` is ``True``

    Raises
    ------
    ValueError
        when `filter_name` is not
        ``"avarage"``, ``"gaussian"``, ``"median"``, ``"gaussian-bilateral"``
        or ``"gaussian-bilateral"
    '''
    if filter_name == 'average':
        logger.info('apply "average" filter')
        smoothed_image = cv2.blur(
            image, (filter_size, filter_size)
        )
    elif filter_name == 'gaussian':
        logger.info('apply "gaussian" filter')
        smoothed_image = mh.gaussian_filter(
            image, sigma=filter_size
        ).astype(image.dtype)
    elif filter_name == 'gaussian-bilateral':
        logger.info('apply "gaussian-bilateral" filter')
        smoothed_image = cv2.bilateralFilter(
            image, filter_size, sigma_color, sigma_space
        )
    elif filter_name == 'median':
        logger.info('apply "median" filter')
        smoothed_image = mh.median_filter(
            image, np.ones((filter_size, filter_size), dtype=image.dtype)
        )
    elif filter_name == 'median-bilateral':
        logger.info('apply "median-bilateral" filter')
        smoothed_image = skimage.filters.rank.mean_bilateral(
            image, skimage.morphology.disk(filter_size),
            s0=sigma_space, s1=sigma_space
        )
    else:
        raise ValueError(
            'Arugment "filter_name" can be one of the following:\n'
            '"average", "gaussian", "median", and "gaussian-bilateral" and '
            '"median-bilateral"'
        )

    output = {'smoothed_image': smoothed_image.astype(image.dtype)}
    if plot:
        logger.info('create plot')
        from jtlib import plotting
        clip_value = np.percentile(image, 99.99)
        data = [
            plotting.create_intensity_image_plot(
                image, 'ul', clip_value=clip_value
            ),
            plotting.create_intensity_image_plot(
                smoothed_image, 'ur', clip_value=clip_value
            ),
        ]
        output['figure'] = plotting.create_figure(
            data,
            title='smoothed with {0} filter (kernel size: {1})'.format(
                filter_name, filter_size
            )
        )
    else:
        output['figure'] = str()

    return output