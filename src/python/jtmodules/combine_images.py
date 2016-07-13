''' combine_images module for TissueMAPS'''
 
import numpy as np

VERSION = '0.1.0'


def combine_images(image01, image02, multiplication_factor01=1,
                   multiplication_factor02=1, plot=False):
    ''' Combine two grayscale images together. The output is a 16-bit grayscale
        image. Note that each image can be multiplied by a factor (positive 
        integer only) prior to combination. Default factor for both images is 1.

        For arithmetics images are cast to type ``float``. This step is made to ensure
        that the resulting intensities values can lie beyond the range of the
        bit depth (>65536 in the case of a 16-bit image).
        
        Each image is then individually multiplied by its respective
        multiplication factor.
        
        The images are then added and the combined image is divided by the sum
        of the two multiplication factors. In consequence the range of the
        combined image is the same as the original images.

        The combined image is finally cast back to its original data type.

        For example, the combine_images module could be used to combine the
        image where lipid droplets were imaged and the image where the cell 
        outline was imaged. The output image can be used as a mask to properly 
        segment adipocytes.

    Parameters
    ----------
    image01: numpy.ndarray[numpy.uint8 or numpy.uint16]
        First Grayscale image that should be combined.
    image02: numpy.ndarray[numpy.uint8 or numpy.uint16]
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

    Author
    ------
    Christophe Freyre
    '''
    #Error types
    if not isinstance(multiplication_factor01, int):
        raise TypeError('Multiplication factor #1 must have integer type.')
    if not isinstance(multiplication_factor02, int):
        raise TypeError('Multiplication factor #2 must have integer type.')
    if multiplication_factor01 >= 1:
        raise ValueError('Multiplication factor #1 must be a positive integer.')
    if multiplication_factor02 >= 1:
        raise ValueError('Multiplication factor #2 must be a positive integer.')
    logger.info(
       'First image will be multiplied by %d', multiplication_factor01
    )
    logger.info(
       'Second image will be multiplied by %d', multiplication_factor02
    )
    
    if image01.shape != image02.shape:
        raise ValueError('The two images must have identical dimensions.')

    #Image Analysis
    # Image are first combined using float type
    weighted_image01 = image01.astype(np.float64) * np.float64(multiplication_factor01)
    weighted_image02 = image02.astype(np.float64) * np.float64(multiplication_factor02)

    # Combined image is cast back to original type
    combined_image = weighted_image01 + weighted_image02
    denominator = np.float64(multiplication_factor01) + np.float64(multiplication_factor02)
    combined_image = combined_image / denominator
    combined_image = combined_image.astype(image01.dtype)

    outputs = {'combined_image': combined_image}

    #Plotting
    if plot:
        from jtlib import plotting
        clip_val = np.percentile(image01, 99.9)
        plots = [
            plotting.create_intensity_image_plot(image01, 'ul', clip=True, clip_val),
            plotting.create_intensity_image_plot(image02, 'ur', clip=True, clip_val),
            plotting.create_intensity_image_plot(combined_image, 'll', clip=True),
        ]
        outputs['figure'] = plotting.create_figure(
            plots, title='Original and combined images'
        )
    else:
        outputs['figure'] = str()

    return outputs
