function [ channel_num ] = channel2num( channel_string )
%channel2num Converts the name of the channel into a number
%   Possible string vals:
% Electrode 1 - Electrode 19
% PDC - Fluid Pressure
% PAC - Fluid Pressure
% Tdc - Temperature
% Tac - Temperate Change

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

electrode_channel = strfind(channel_string, 'Electrode');

if (numel(electrode_channel) > 0)
    channel_num = str2double(channel_string(11:end));
else
    switch (channel_string)
        case 'PDC - Fluid Pressure'
            channel_num = 20;
        case 'PAC - Fluid Pressure'
            channel_num = 23;
        case 'Tdc - Temperature'
            channel_num = 22;
        case 'Tac - Temperate Change'
            channel_num = 21;
        otherwise
            disp ('Could not locate channel')
    end
end

end

