function [ data_separated ] = separate_states( data, fix )
%separate_states Given a dataset, split it according to the control states

states = unique(data.controller_state);
data_separated = struct();
biotac_constant_fields = [{'finger'}, {'bt_serial'}];
biotac_pac_fields = [{'raw_pac_flat'}, {'pac_flat'}];

for i = 1:size(states,2)
    state = states(i);
    
    % Get the index of that state
    idx = (data.controller_state == state);
    
    if (fix)
        start_idx = find(idx, 1, 'first');
        idx = circshift(idx, [0 50]);
        idx(1:start_idx) = 0;
    end
    
    % Store segment
    data_segment = struct();
    
    data_segment.transforms = data.transforms(:, idx);
    data_segment.joint_states = data.joint_states(:, idx);
    data_segment.gripper_accelerometer = data.gripper_accelerometer(:, idx);
    biotac = data.biotac;
    
    biotac_store = [];
    % Go through the biotac fields
    for j = 1:size(biotac,2)
        biotac_finger = struct();
        biotac_fields = fields(biotac(j));
        for k = 1:size(biotac_fields)
                field_name = biotac_fields{k};
                values = biotac(j).(field_name);
                
                % Check if we need to split or not
                if (ismember(field_name, biotac_constant_fields))
                    biotac_finger.(field_name) = values;
                elseif (ismember(field_name, biotac_pac_fields))
                    % Do nothing if it is a flat variable
                else
                    biotac_finger.(field_name) = values(idx,:,:);
                end
        end
        biotac_finger.pac_flat = reshape(biotac_finger.pac',[],1);
        biotac_finger.raw_pac_flat = reshape(biotac_finger.raw_pac',[],1);
        biotac_store = [biotac_store biotac_finger];
    end
    
    data_segment.biotac = biotac_store;
                
    % Split data with idx
    switch(state)
        case 0
            data_separated.start = data_segment;
        case 1
            data_separated.hold = data_segment;
        case 2
            data_separated.slide = data_segment;
        case 3
            data_separated.squeeze = data_segment;
        case 5
            data_separated.tap = data_segment;
        case 6
            data_separated.slide_fast = data_segment;
    end
end
            


end

