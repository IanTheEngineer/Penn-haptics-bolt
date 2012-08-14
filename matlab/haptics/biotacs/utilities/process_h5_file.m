function [ all_data ] = process_h5_file( h5_file_path )
%process_h5_file Given a h5 file, loads and parses

% Check if PC or linux
file_concat = '/';

% Open h5 file
info = h5info(h5_file_path);

loc = H5F.open(h5_file_path, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');

% Pull out each file as a struct
file_groups = info.Groups;

all_data = struct('data',{}, 'name', {});

for file_num = 1:size(file_groups,1)
    
    % pull file out of the root
    file = file_groups(file_num);
    file_name = file.Name;
    file_loc = H5G.open(loc, file_name); % actually open file
    all_data(file_num).name = file_name;
    
    % Create name to search for biotacs
    biotac_name = strcat(file_name, file_concat, 'biotacs');
    % Pull biotac location
    biotac_loc = H5G.open(file_loc, biotac_name);
    
    % Find the group location of the biotacs and store
    idx = find(ismember({file.Groups.Name},biotac_name)==1);
    biotacs = file.Groups(idx);
    
    % Structure to store biotac info pulled out
    biotac_data = struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});
    
    % For each finger pull information
    for num_count = 0:size(biotacs.Groups)-1
        
        finger = num_count + 1;
        
        % Find the finger inside the biotac group
        finger_name = strcat(biotac_name, file_concat, 'finger_', int2str(num_count));       
        finger_loc = H5G.open(biotac_loc, finger_name);
      
        % Pull out electrodes
        electrodes_loc = H5D.open(finger_loc, 'electrodes');
        electrodes = H5D.read(electrodes_loc);
        
        % Pull out PDC
        pdc_loc = H5D.open(finger_loc, 'pdc');
        pdc = H5D.read(pdc_loc);
        
        % Pull out PAC
        pac_loc = H5D.open(finger_loc, 'pac');
        pac = H5D.read(pac_loc);
        
        % Pull out TDC
        tdc_loc = H5D.open(finger_loc, 'tdc');
        tdc = H5D.read(tdc_loc);
        
        % Pull out TAC
        tac_loc = H5D.open(finger_loc, 'tac');
        tac = H5D.read(tac_loc);
        
        % Store and process data
        biotac_data(finger).raw_electrodes = double(electrodes)';
        biotac_data(finger).raw_pdc = double(pdc);
        biotac_data(finger).raw_tac = double(tac);
        biotac_data(finger).raw_tdc = double(tdc);
        biotac_data(finger).raw_pac = double(pac)';
        biotac_data(finger).raw_pac_flat = double(reshape(pac,[],1));
        
        % Processed data
        biotac_data(finger).electrodes = -bsxfun(@minus, biotac_data(finger).raw_electrodes,mean(biotac_data(finger).raw_electrodes(1:10,:)));
        biotac_data(finger).pdc = biotac_data(finger).raw_pdc - mean(biotac_data(finger).raw_pdc(1:10));
        % Add a negative because Syntouch's filter inverts the TAC signal
        biotac_data(finger).tac = -(biotac_data(finger).raw_tac - mean(biotac_data(finger).raw_tac(1:10)));
        biotac_data(finger).tdc =  biotac_data(finger).raw_tdc - mean(biotac_data(finger).raw_tdc(1:10));
        % Add a negative because Syntouch's filter inverts the PAC signal
        biotac_data(finger).pac = -bsxfun(@minus, biotac_data(finger).raw_pac,mean(biotac_data(finger).raw_pac(1:10,:)));
        biotac_data(finger).pac_flat = -(biotac_data(finger).raw_pac_flat - mean(biotac_data(finger).raw_pac_flat(1:10)));

        % Convert values into RGB values
        cur_channel_values = -double(biotac_data(finger).electrodes);
        num_colors = size(unique(cur_channel_values),1);
        parsed_color = ind2rgb(gray2ind(mat2gray(cur_channel_values),num_colors),autumn(num_colors));
        biotac_data(finger).color = parsed_color;  
        
    end
        
    % Pull out accelerometer information
    accelerometer_loc = H5D.open(file_loc, 'accelerometer');
    accelerometer_data = H5D.read(accelerometer_loc);
    
    % Pull out gripper aperture information
    % Structure to store gripper_aperture info pulled out
    gripper_data = struct('joint_effort', {},'joint_position',{},'joint_velocity',{});
    
    % Pull out gripper_aperture information
    gripper_aperture_name = strcat(file_name, file_concat, 'gripper_aperture');
    gripper_aperture_loc = H5G.open(file_loc, gripper_aperture_name);
    
    % Pull out gripper joint effort
    gripper_joint_effort_loc = H5D.open(gripper_aperture_loc, 'joint_effort');
    gripper_data(1).joint_effort = H5D.read(gripper_joint_effort_loc);

    % Pull out gripper joint position
    gripper_joint_position_loc = H5D.open(gripper_aperture_loc, 'joint_position');
    gripper_data(1).joint_position = H5D.read(gripper_joint_position_loc);
        
    % Pull out gripper joint_velocity
    gripper_joint_velocity_loc = H5D.open(gripper_aperture_loc, 'joint_velocity');
    gripper_data(1).joint_velocity = H5D.read(gripper_joint_velocity_loc);

    % Pull out state information
    % Structure to store gripper_aperture info pulled out
    state_data = struct('controller_state', {},'controller_state_detail',{});

    % Pull out gripper_aperture information
    controller_state_name = strcat(file_name, file_concat, 'state');
    controller_state_loc = H5G.open(file_loc, controller_state_name);
    
    % Regular state
    state_loc = H5D.open(controller_state_loc, 'controller_state');
    state_data(1).controller_state = double(H5D.read(state_loc));
    
    % Detailed state
    state_detail_loc = H5D.open(controller_state_loc, 'controller_detail_state');
    state_data(1).controller_state_detail = H5D.read(state_detail_loc);
    
    % Datastructure to store all data
    all_data_robot = struct('biotac',{},'accelerometer',{} , 'controller_state', {},'gripper_aperture',{});
    
    % Store biotac information
    all_data_robot(1).biotac = biotac_data;
    
    % Store acceleroometer
    all_data_robot(1).accelerometer = accelerometer_data;
    
    % Store gripper aperture information
    all_data_robot(1).gripper_aperture = gripper_data;
    
    % Store state information
    all_data_robot(1).controller_state = state_data;
    
    % Store each file
    all_data(file_num).data = all_data_robot;
    
end

