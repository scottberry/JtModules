'''Jterator module for combining two images into one.'''
import numpy as np
import skimage.util

VERSION = '0.0.1'


def main(image01, image02, factor01=1, factor02=1, plot=False):
    '''Combine by a factor prior to combination.

    Parameters
    ----------
    image01: numpy.ndarray[numpy.uint8 or numpy.uint16]
        first grayscale image
    image02: numpy.ndarray[numpy.uint8 or numpy.uint16]
        second grayscale image
    factor01: float, optional
        value by which `image01` should be multiplied (default: ``1``)
    factor02: float, optional
        value by which `image02` should be multiplied (default: ``1``)
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    Dict[str, numpy.ndarray[bool] or str]
        * "combined_image": resulting combined image
        * "figure": JSON string figure representation

    Note
    ----
    The product of `factor01` and `factor02` must equal ``1``.

    Raises
    ------
    ValueError
        when the two image arrays to combined do not have the same shape
    '''
    logger.info('first image will be multiplied by %d', factor01)
    logger.info('second image will be multiplied by %d', factor02)

    if image01.dtype != image02.dtype:
        raise ValueError('Images must have the same data type.')
    if image01.shape != image02.shape:
        raise ValueError('Images must have the same dimensions.')
    if (factor01 * factor02) != 1:
        raise ValueError('The product of factors must equal one.')

    # Map images through a lookup table to normalize intensity values and
    # multiply them with the given factors.
    limits = skimage.util.dtype_limits(image)
    lut = np.linspace(0, 1, limits[1]+1)
    weighted_image01 = lut[image01] * factor01
    weighted_image02 = lut[image02] * factor02

    # Add images and cast them back to original data type
    combined_image = np.add(weighted_image01, weighted_image02)
    combined_image = (combined_image * limits[1]).astype(image01.dtype)

    outputs = {'combinedimage': combined_image}
    if plot:
        from jtlib import plotting
        plots = [
            plotting.create_intensity_image_plot(image01, 'ul'),
            plotting.create_intensity_image_plot(image02, 'ur'),
            plotting.create_intensity_image_plot(combined_image, 'll'),
        ]
        outputs['figure'] = plotting.create_figure(
            plots, title='Individual images and their combination'
        )
    else:
        outputs['figure'] = str()

    return outputs
