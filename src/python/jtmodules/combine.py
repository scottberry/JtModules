''' Combine module for TissueMAPS'''
import numpy as np

VERSION = '0.0.1'


def main(image01, image02,multiplication_factor01,multiplication_factor02, plot=False):
    ''' Combine two images together. Note that each image can be multiplied by a factor prior to combination

    Parameters
    ----------
    image01: numpy.ndarray
        First Grayscale image that should be combined.
    image02: numpy.ndarray
        Second Grayscale image that should be combined.
    multiplication_factor01: int, optional
        value by which the image01 should be multiplied
        (default: ``1``)
    multiplication_factor02: int, optional
        value by which the image02 should be multiplied
        (default: ``1``)
    plot: bool, optional
        whether a plot should be generated (default: ``False``)
    Returns
    -------
    Dict[str, numpy.ndarray[bool] or str]
        * "combined_image":  resulting combined image
        * "figure": html string in case `plot` is ``True``
    Raises
    ------
    ValueError
        when the two images to combined do not have the same dimensions (NxM)
    '''
    '''Error types'''
    if multiplication_factor01 is None:
        multiplication_factor01 = 1
    logger.info('First image will be multiplied by 1')
    if multiplication_factor02 is None:
        multiplication_factor02 = 1
    logger.info('Second image will be multiplied by 1')
    
    if image01.shape != image02.shape:
        raise ValueError('Images do not have the same dimension. Please selected images which have the same dimensions')
    
    '''Image Analysis'''
    weightedImage01 = np.multiply(image01,multiplication_factor01)
    weightedImage02 = np.multiply(image02,multiplication_factor02)
    
    addedImages = np.zeros_like(weightedImage01,dtype=none,order='K',subok=True)
    addedImages = np.add(weightedImage01,weightedImage02)
    combinedImage = np.add(weightedImage01,weightedImage02)
    outputs = {'combinedimage': combinedimage}
    
    '''Plotting'''
    if plot:
        from jtlib import plotting

        plots = [
            plotting.create_overlay_image_plot(image, combined_image, 'ul'),
            plotting.create_mask_image_plot(ombined_image, 'ur'),
            [
                plotting.create_histogram_plot(image.flatten(), 'll'),
                plotting.create_line_plot(
                        [0, np.prod(image.shape)/100],
                        [corr_thresh, corr_thresh],
                        'll',
                        color=plotting.OBJECT_COLOR, line_width=4,
                    )
            ]
        ]
        '''Figures'''
        outputs['figure'] = plotting.create_figure(
                        plots, plot_is_image=[True, True, False],
                        title='''Combined image at pixel value %s
                        ''' % thresh
        )
    else:
        outputs['figure'] = str()

    return outputs
