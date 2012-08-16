function [ all_data, number_fingers ] = process_txt_file( fileID )
%process_txt_file Given the output format from Syntouch's default logger,
%reads and processes into proper data structure

% Input: .txt file from Syntouch loggers (either C library or LabView)
% Output: structure of format: struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});
% Multiple fingers are in an array of structs.

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

% Configurable Values (Mainly for the format of the Syntouch Logger)
finger_index = [5 7 9];
finger_channels = [];
num_values_frame = 44;
num_frames_throw_away = 1;

data = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f');
fclose(fileID);

% Selects the proper finger channel
if (data{6}(1) == 0)
    finger_channels = [finger_channels 1];
elseif (data{8}(1) == 0)
    finger_channels = [finger_channels 2];
elseif (data{10}(1) == 0)
    finger_channels = [finger_channels 3];
end

number_fingers = size(finger_channels,2);

all_data = struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});
for i = 1:number_fingers
    all_values = data{finger_index(finger_channels(i))};
    values = reshape(all_values, num_values_frame,numel(all_values)/num_values_frame)';

    % Electrodes are every other frame, also throws away frames in the
    % beginning of the data file
    all_data(i).raw_electrodes = values(num_frames_throw_away:end,1:2:end-6);
    all_data(i).raw_pdc = values(num_frames_throw_away:end,end-5:2:end-4);
    all_data(i).raw_tac = values(num_frames_throw_away:end,end-3:2:end-2);
    all_data(i).raw_tdc = values(num_frames_throw_away:end,end-1:2:end);
    all_data(i).raw_pac = values(num_frames_throw_away:end,2:2:end);
    % Add a negative because Syntouch's filter inverts the PAC signal
    all_data(i).raw_pac_flat = reshape(all_data(i).raw_pac',[],1);

    % Processed Data
    all_data(i).electrodes = -bsxfun(@minus, all_data(i).raw_electrodes,mean(all_data(i).raw_electrodes(1:10,:)));
    all_data(i).pdc = all_data(i).raw_pdc - mean(all_data(i).raw_pdc(1:10));
    % Add a negative because Syntouch's filter inverts the TAC signal
    all_data(i).tac = -(all_data(i).raw_tac - mean(all_data(i).raw_tac(1:10)));
    all_data(i).tdc =  all_data(i).raw_tdc - mean(all_data(i).raw_tdc(1:10));
    % Add a negative because Syntouch's filter inverts the PAC signal
    all_data(i).pac = -bsxfun(@minus, all_data(i).raw_pac,mean(all_data(i).raw_pac(1:10,:)));
    all_data(i).pac_flat = -(all_data(i).raw_pac_flat - mean(all_data(i).raw_pac_flat(1:10)));

    all_data(i).finger = finger_index(finger_channels(i));

    % Convert values into RGB values
    cur_channel_values = -double(all_data(i).electrodes);
    num_colors = size(unique(cur_channel_values),1);
    parsed_color = ind2rgb(gray2ind(mat2gray(cur_channel_values),num_colors),autumn(num_colors));
    all_data(i).color = parsed_color;
end    

end

