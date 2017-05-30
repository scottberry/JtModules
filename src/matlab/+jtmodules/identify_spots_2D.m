% Copyright 2017 Scott Berry, University of Zurich
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

classdef identify_spots_2D
    properties (Constant = true)

        VERSION = '0.0.1'

    end

    methods (Static)

        function [spots, spots_deblend, figure] = main(image, spot_size, rescale_quantile_min, rescale_quantile_max, min_of_min, max_of_min, min_of_max, max_of_max, detection_threshold, deblending_steps, plot, plot_clip_value)

%            import jtlib.plotting;
%            import cpsub.*;

            if min_of_min == 0; min_of_min = NaN; end;
            if max_of_min == 0; max_of_min = NaN; end;
            if min_of_max == 0; min_of_max = NaN; end;
            if max_of_max == 0; max_of_max = NaN; end;

            min_of_min = NaN;
            max_of_min = 120;
            min_of_max = 500;
            max_of_max = NaN;

            Options.ObSize = double(spot_size);
            Options.limQuant = double([rescale_quantile_min rescale_quantile_max]);
            Options.RescaleThr = [min_of_min max_of_min min_of_max max_of_max];
            Options.ObjIntensityThr = [];
            Options.closeHoles = false;
            Options.ObjSizeThr = [];
            Options.ObjThr = detection_threshold;
            Options.StepNumber = deblending_steps;
            Options.numRatio = 0.20;
            Options.doLog = 0;
            Options.DetectBias = [];

            %error('%i, %i', Options.RescaleThr(3),Options.RescaleThr(4))

            log_filter = cpsub.fspecialCP3D('2D LoG',Options.ObSize);

            [ObjCount{1} SegmentationCC{1} FiltImage] = cpsub.ObjByFilter(double(image),log_filter,Options.ObjThr,Options.limQuant,Options.RescaleThr,Options.ObjIntensityThr,true,[],Options.DetectBias)
            spots = int32(labelmatrix(SegmentationCC{1}));

            % Deblend objects
            if deblending_steps > 0
                spots_deblend = int32(cpsub.SourceExtractorDeblend(double(image),SegmentationCC{1},FiltImage,Options));
            end

            output_image = image;

            if nargin < 2
                plot = false;
            end

            if plot
                figure = jtlib.plotting.create_intensity_image_plot(output_image, 'ul');
            else
                figure = '';
            end

        end

    end
end
