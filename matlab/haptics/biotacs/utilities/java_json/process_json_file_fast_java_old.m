function [ all_data, number_fingers ] = process_json_file_fast( fileID )
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
num_frames_throw_away = 50;

% Read data from JSON file
data_read = [];

rline = fgetl(fileID); % '['
if (strcmp(rline(1),'['))
    rline = fgetl(fileID);
end

while (ischar(rline) && ~strcmp(rline(1),']'))
    data_read = [data_read net.minidev.json.JSONValue.parse(rline)];
    rline = fgetl(fileID);
end
fclose(fileID);

% Checks if the JSON file contains transforms information
if isempty(data_read(1).get('bt_hand'))
    pr2_transforms = 0;
    data = data_read;
else
    pr2_transforms = 1;
    data = cell2mat(arrayfun(@(x) x.get('bt_hand'),data_read,'UniformOutput', false));
end
    
number_fingers = size(data(1).get('bt_data'));

% Pull out information from java object
bt_data = [];
for num = 1:size(data)
    bt_object = data(num).get('bt_data');
    temp_store = [];
    for num_fingers = 0:size(bt_object)-1
        temp_store = [temp_store bt_object.get(num_fingers)];
    end
    bt_data = [bt_data; temp_store];
end

% Reshape time information
%bt_time = [data(:).bt_time];

all_data = struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});

% Go through all of the json objects
for i = 1:number_fingers
    
    % Pull out the data for the specific finger
    values = bt_data(:,i);
    
    % Go through each object (need loop because it is stored as a java
    % object)
    raw_electrodes = zeros(size(values,1),19);
    raw_pac = zeros(size(values,1),22);
    raw_pdc = zeros(size(values));
    raw_tdc = zeros(size(values));
    raw_tac = zeros(size(values));
    for j = 1:size(values,1)
        % For electrodes
        store_array = values(j).get('electrode_data');
        raw_electrodes(j,:) = cell2mat(store_array.toArray.cell);
        
        % for pac data
        store_array = values(j).get('pac_data');
        raw_pac(j,:) = cell2mat(store_array.toArray.cell);
        
        % Store PDC values
        raw_pdc(j) = values(j).get('pdc_data');
        raw_tac(j) = values(j).get('tac_data');
        raw_tdc(j) = values(j).get('tdc_data');
    end
    
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

    all_data(i).finger = values(1).get('bt_position');
    all_data(i).bt_serial = values(1).get('bt_serial');

    % Convert values into RGB values
    cur_channel_values = -double(all_data(i).electrodes);
    num_colors = size(unique(cur_channel_values),1);
    parsed_color = ind2rgb(gray2ind(mat2gray(cur_channel_values),num_colors),autumn(num_colors));
    all_data(i).color = parsed_color;
end    

toc

end

