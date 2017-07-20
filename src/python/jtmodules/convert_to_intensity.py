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
'''Jterator module converting a label_image to an intensity_image.'''
import collections
import logging
import numpy as np

VERSION = '0.0.1'

Output = collections.namedtuple('Output', ['intensity_image', 'figure'])

logger = logging.getLogger(__name__)


def main(label_image, plot=False):
    '''Converts a LabelImage to an IntensityImage

    Parameters
    ----------
    label_image: numpy.ndarray[numpy.int32]
        label image to be converted
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    jtmodules.convert_to_intensity.Output
    '''
    logger.info('Converting label image to intensity image')
    if (np.amax(label_image) < pow(2, 16)):
        intensity_image = label_image.astype(dtype=np.uint16)
    else:
        logger.warn(
            '%d objects in input label image exceeds maximum (%d)',
            np.amax(label_image),
            pow(2, 16)
        )
        intensity_image = label_image

    if plot:
        from jtlib import plotting
        n_objects = len(np.unique(label_image)[1:])
        colorscale = plotting.create_colorscale(
            'Spectral', n=n_objects, permute=True, add_background=True
        )
        plots = [
            plotting.create_mask_image_plot(
                label_image, 'ul', colorscale=colorscale
            ),
            plotting.create_intensity_image_plot(
                intensity_image, 'ur'
            )
        ]
        figure = plotting.create_figure(plots, title='convert_to_intensity_image')
    else:
        figure = str()

    return Output(intensity_image, figure)
