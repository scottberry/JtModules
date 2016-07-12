import mahotas as mh
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.measure import regionprops
from scipy import ndimage as ndi
import numpy as np
import skimage as sk
from skimage.morphology import watershed, binary_dilation
from matplotlib import pyplot as plt


def main(input_label_image, input_image, threshold_correction_factors,
        min_threshold, plot=False):

    if np.any(input_label_image == 0):
        has_background = True
    else:
        has_background = False

    # Calculate the values for the different thresholds levels by
    # multiplying an initial threshold value with the provided correction factors
    n_thresholds = len(threshold_correction_factors)
    # Convert to floating points for arithmetics
    thresholds = np.zeros(n_thresholds, np.float64)
    thresholds[0] = np.float64(mh.otsu(input_image))
    for i in range(1, n_thresholds):
        corrected_thresh = (
            thresholds[0] * threshold_correction_factors[i] /
            threshold_correction_factors[0]
        )
        if corrected_thresh < min_threshold:
            corrected_thresh = min_threshold
        thresholds[i] = corrected_thresh

    # Remove duplicates and sort in descending order
    thresholds = np.unique(thresholds)[::-1]
    # Cast back to original data type
    thresholds = thresholds.astype(input_image.dtype)
    n_thresholds = len(thresholds)

    primary_object_mask = input_label_image > 0
    primary_object_outlines = mh.labeled.bwperim(primary_object_mask)

    # Two watershed transformations are applied at each threshold level.
    # The first is applied to a gradient image (intensity image filtered with
    # Sobel filter) and a second to the intensity image directly.
    # We apply the Sobel filter twice, once horizontally and once vertically
    # and then integrate the information of both. We take the absolute gradient
    # values.
    weights = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    sobel_image_h = mh.convolve(input_image.astype(float), weights)
    sobel_image_v = mh.convolve(input_image.astype(float), weights.T)
    gradient_image = np.abs(sobel_image_v) + np.abs(sobel_image_h)
    gradient_image = gradient_image.astype(input_image.dtype)

    label_image_stack = np.zeros(
        input_label_image.shape + (n_thresholds,), input_label_image.dtype
    )
    if not has_background:
        output_label_image = input_label_image
    else:
        for k in range(1, n_thresholds):
            print k

            def _perform_watershed(int_img, mask, thresh, ws_img):
                threshold_mask = int_img > thresh
                marker_mask = np.logical_or(mask, ~threshold_mask)
                outlines = mh.labeled.bwperim(mask)
                marker_mask[outlines] = False
                # TODO: how to implement MATLAB's imimposemin?
                # This is not the correct implementation, but it may still do
                # the trick.
                peaks = mh.dilate(ws_img)
                peaks[marker_mask] = 0
                se = np.ones((3, 3), bool)
                seeds = mh.label(peaks, Bc=se)[0]
                watershed_regions = mh.cwatershed(peaks, seeds)

                actual_objects = watershed_regions.copy()
                # Ensure objects are separated
                watershed_lines = mh.thin(mh.labeled.borders(watershed_regions))
                actual_objects[watershed_lines] = 0
                # Remove "background" regions, i.e. watershed regions not
                # overlapping any primary objects
                correct_region_ids = actual_objects[mask]
                lut = np.zeros(np.max(actual_objects)+1, actual_objects.dtype)
                lut[correct_region_ids] = correct_region_ids
                import ipdb; ipdb.set_trace()
                return lut[actual_objects]

            # plt.imshow(gradient_image); plt.show()
            # import ipdb; ipdb.set_trace()
            pre_secondary_object_mask = _perform_watershed(
                input_image, primary_object_mask, thresholds[k],
                gradient_image
            )
            import ipdb;ipdb.set_trace()

            secondary_object_mask = _perform_watershed(
                input_image, pre_secondary_object_mask, thresholds[k],
                np.invert(input_image)
            )
            secondary_object_mask = mh.close_holes(secondary_object_mask)

            # Label secondary objects according to primary objects
            se = np.ones((3, 3), bool)
            labels = mh.label(secondary_object_mask, Bc=se)[0]
            primary_object_ids = np.unique(input_label_image)[1:]
            for i in primary_object_ids:
                mask = input_label_image == i
                oid = labels[mask]
                if oid.size == 0:
                    # Ensure that every primary object has a secondary object.
                    # In case no secondary object could be identified, use
                    # the (dilated) primary object.
                    index = mh.morph.dilate(primary_object_mask)
                else:
                    index = labels == oid
                label_image_stack[:, :, k][index] = i

            # Safety first
            final_label_image[:, :, k][primary_object_mask] = \
                input_label_image[primary_object_mask]


        # output_label_image = np.max(label_images_stack, axis=2)
        output_label_image = np.zeros(
            input_label_image.shape, input_label_image.dtype
        )
        for i in range(n_thresholds):
            plane = label_image_stack[:, :, -i]
            output_label_image[plane > 0] = plane[plane > 0]

    output = dict()
    output['output_label_image'] = output_label_image

    if plot:
        from jtlib import plotting
        n_objects = len(np.unique(output_label_image)[1:])
        colorscale = plotting.create_colorscale(
            'Spectral', n=n_objects, permute=True, add_background=True
        )
        plots = [
            plotting.create_mask_image_plot(
                input_label_image, 'ul', colorscale=colorscale
                ),
            plotting.create_mask_image_plot(
                output_label_image, 'ur', colorscale=colorscale
            ),
            plotting.create_overlay_image_plot(
                input_image, output_label_image, 'll'
            )
        ]
        output['figure'] = plotting.create_figure(
            plots, title='secondary objects'
        )
    else:
        output['figure'] = str()

    return output


if __name__ == '__main__':

    import cv2
    from matplotlib import pyplot as plt

    threshold_correction_factors = [
        2, 1.5, 1.3, 0.9, 0.7, 0.6, 0.58, 0.55, 0.50, 0.45, 0.4, 0.35, 0.3, 0.25
    ]
    min_threshold = 120

    mask = cv2.imread('/tmp/mask.tif', cv2.IMREAD_UNCHANGED)
    img = cv2.imread('/tmp/celltrace.tif', cv2.IMREAD_UNCHANGED)

    main(mask, img, threshold_correction_factors, min_threshold)


