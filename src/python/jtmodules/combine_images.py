''' combine_images module for TissueMAPS'''
 
import numpy as np

VERSION = '0.1.0'


def combine_images(image01, image02,multiplication_factor01,
                   multiplication_factor02, plot=False):
    ''' Combine two grayscale images together. The output is a 16-bit grayscale
        image. Note that each image can be multiplied by a factor (positive 
        integer only) prior to combination. Default factor for both images is 1.

        Each image is first converted in uint32. This step is made to ensure
        that the resulting intensities values can lie beyond the range of the
        image (>65536 in the case of a 16-bit image).
        
        Each image is then individually multiplied by its respective
        multiplication factor.
        
        The images are combined together and divided by the sum of the two 
        multiplication factors. In consequence the range of the combined image 
        is the same as the original images.

        The combined image is finally converted to a uint16 (16-bit image).

        For example, the combine_images module could be used to combine the
        image where lipid droplets were imaged and the image where the cell 
        outline was imaged. The output image can be used as a mask to properly 
        segment adipocytes.

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

    Author: Christophe Freyre
    Contact: christophe.freyre (at) uzh (dot) ch 
    Date: 12.07.2016
    '''
    #Error types
    if multiplication_factor01 is None:
        multiplication_factor01 = 1
    logger.info('First image will be multiplied by 1')
    if multiplication_factor02 is None:
        multiplication_factor02 = 1
    logger.info('Second image will be multiplied by 1')
    
    if image01.shape != image02.shape:

    if np.less(multiplication_factor01,1):
        raise ValueError('Multiplication factor must be a positive integer')

    if np.less(multiplication_factor02,1):
        raise ValueError('Multiplication factor must be a positive integer')
    
    #Image Analysis
    # Image are first converted to unint32 prior to multiplication
    image01 = np.uint32(image01)
    image02 = np.uint32(image02)
    weighted_image01 = np.zeros_like(image01,dtype=np.uint32)
    weighted_image02 = np.zeros_like(image02,dtype=np.uint32)  
    weighted_image01 = np.multiply(image01,multiplication_factor01)
    weighted_image02 = np.multiply(image02,multiplication_factor02)

    # Could add dtype=np.uint16 below to force the output to be a 16-bit image.
    added_images = np.zeros_like(weighted_image01,dtype=np.uint32) 
    added_images = np.add(weighted_image01,weighted_image02)
    denominator = np.add(multiplication_factor01,multiplication_factor02)
    combined_image = np.divide(added_images,denominator)
    combined_image = np.uint16(combined_image)
    outputs = {'combined_image': combined_image}

    #Plotting
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
        #Figures
        outputs['figure'] = plotting.create_figure(
                        plots, plot_is_image=[True, True, False],
                        title='''Combined image at pixel value %s
                        ''' % thresh
        )
    else:
        outputs['figure'] = str()

    return outputs
