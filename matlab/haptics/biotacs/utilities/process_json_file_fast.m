function [ all_data_robot, number_fingers ] = process_json_file_fast( fileID )
%process_json_structure Given a path to a file that is known to be a JSON
%file.  This will return a structure of all the data processed.

% Input: From the biotac_stack ROS stack.  The two types include
% just BioTac information, and BioTacs with PR2 transforms.
% Output: structure of format: struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});
% Multiple fingers are in an array of structs.

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012
tic

% Add JSON-c library to path
addpath(genpath('json_c'));

% For how many frames to ignore
num_frames_throw_away = 50;

% Read data from JSON file
data_read = [];
rline = fgetl(fileID); % '['
if (strcmp(rline(1),'['))
    rline = fgetl(fileID);
end
i = 1;

while (ischar(rline) && ~strcmp(rline(1),']'))
    data_read = [data_read fromjson(sprintf(rline))];
    i = i+1;
    rline = fgetl(fileID);
end
toc
fclose(fileID);

% Checks if the JSON file contains transforms information
if (isfield(data_read, 'bt_hand'))
    % Process data according to PR2
    pr2_transforms = 1;
    data = [data_read(:).bt_hand];
else
    pr2_transforms = 0;
    data = data_read;
end

number_fingers = size(data(1).bt_data,2);

% Reshape data
bt_data = [data(:).bt_data];
%bt_data = reshape(bt_data, number_fingers, size(data,2));

% Reshape time information
bt_time = [data(:).bt_time];

all_data = struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});

% Go through all of the json objects

for i = 1:number_fingers

    % Pull out the data for the specific finger
    values = [bt_data{i,:}];

    % Pull out electrode values
    electrodes = [values(:).electrode_data];
    electrodes = cell2mat(electrodes);

    % Pull out the AC pressure
    pac_store = [values(:).pac_data];
    pac_store = cell2mat(pac_store);

    % Raw sensor values
    raw_electrodes = reshape(electrodes, 19, size(bt_data,2))';
    raw_pdc = [values(:).pdc_data]';
    raw_tac = [values(:).tac_data]';
    raw_tdc = [values(:).tdc_data]';
    raw_pac = reshape(pac_store,22,size(bt_data,2))';

    % Clean the first few values
    all_data(i).raw_electrodes = raw_electrodes(num_frames_throw_away:end,:);
    all_data(i).raw_pdc = raw_pdc(num_frames_throw_away:end);
    all_data(i).raw_tac = raw_tac(num_frames_throw_away:end);
    all_data(i).raw_tdc = raw_tdc(num_frames_throw_away:end);
    all_data(i).raw_pac = raw_pac(num_frames_throw_away:end,:);
    % Add a negative because Syntouch's filter inverts the PAC signal
    all_data(i).raw_pac_flat = -reshape(all_data(i).raw_pac',[],1);

    % Processed Data
    all_data(i).electrodes = -bsxfun(@minus, all_data(i).raw_electrodes,mean(all_data(i).raw_electrodes(1:10,:)));
    all_data(i).pdc = all_data(i).raw_pdc - mean(all_data(i).raw_pdc(1:10));
    % Add a negative because Syntouch's filter inverts the TAC signal
    all_data(i).tac = -(all_data(i).raw_tac - mean(all_data(i).raw_tac(1:10)));
    all_data(i).tdc =  all_data(i).raw_tdc - mean(all_data(i).raw_tdc(1:10));
    % Add a negative because Syntouch's filter inverts the PAC signal
    all_data(i).pac = -bsxfun(@minus, all_data(i).raw_pac,mean(all_data(i).raw_pac(1:10,:)));
    all_data(i).pac_flat = all_data(i).raw_pac_flat - mean(all_data(i).raw_pac_flat(1:10));

    all_data(i).finger = values(1).bt_position;
    all_data(i).bt_serial = values(1).bt_serial;

    % Convert values into RGB values
    cur_channel_values = -double(all_data(i).electrodes);
    num_colors = size(unique(cur_channel_values),1);
    parsed_color = ind2rgb(gray2ind(mat2gray(cur_channel_values),num_colors),autumn(num_colors));
    all_data(i).color = parsed_color;
end    

if (pr2_transforms)
    all_data_robot = struct('biotac',{},'transforms',{},'joint_states',{}, 'contoller_state', {},'gripper_accelerometer',{});
    
    % Store biotac infor
    all_data_robot(1).biotac = all_data;
    
    % Store transforms, etc.
    num_transforms = size(data_read(1).transforms,2);
    transforms =  cell2mat([data_read(:).transforms]);
    all_data_robot(1).transforms = reshape(transforms, [num_transforms size(transforms,2)/num_transforms]);
    
    % Store joints
    num_joints = size(data_read(1).joint_states,2);
    joints = cell2mat([data_read(:).joint_states]);
    all_data_robot(1).joint_states = reshape(joints, [num_joints size(joints,2)/num_joints]);
    
    % Store controller state
    temp = [data_read(:).controller_state];
    all_data_robot(1).controller_state = [temp(:).data];
    
    % Store gripper_accelerometer
    all_data_robot(1).gripper_accelerometer = [data_read(:).gripper_accelerometer];
else
    all_data_robot = all_data;
end
    
toc
end

