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
'''Jterator module for applying a mask to a label image
'''
import logging
import collections
import mahotas as mh

VERSION = '0.1.0'

logger = logging.getLogger(__name__)

Output = collections.namedtuple('Output', ['masked_image', 'figure'])


def main(objects, mask, plot=False):
    '''Applys a mask to BinaryImage or LabelImage,
    to remove objects outside the mask.

    Parameters
    ----------
    objects: numpy.ndarray[numpy.int32]
        label image or binary image containing objects to be masked
    mask: numpy.ndarray[numpy.int32]
        label image or binary image that should be used as a mask
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    jtmodules.apply_mask.Output[Union[numpy.ndarray, str]]

    '''

    objects[mask == 0] = 0
    mh.labeled.relabel(objects, inplace=True)

    if plot:
        logger.info('create plot')
        from jtlib import plotting
        colorscale = plotting.create_colorscale(
            'Spectral', n=objects.max(), permute=True, add_background=True
        )
        data = [
            plotting.create_mask_image_plot(
                mask, 'ul', colorscale=colorscale
            ),
            plotting.create_mask_image_plot(
                objects, 'ur', colorscale=colorscale
            )
        ]
        figure = plotting.create_figure(
            data,
            title='Masked image with "{0}" objects'.format(
                objects.max()
            )
        )
    else:
        figure = str()

    return Output(objects, figure)
