function [ selected_data ] = select_motion( data, motion_select )
%select_motion Given a motion as a string, selects only the parts of data that contain
%that motion. Data is the specific structure loaded from load_all_data.

% To use - give data as the structure created from load_all_data and
% motion_select as a string.
% Ex. select_motion(data, 'poke') - returns all file names that were
% labeled with poke

data_flat = [];
for i = 1:numel(data)
    data_flat = [data_flat; data(i).data(:)];
end

idx = ismember([data_flat(:).motion],motion_select);

selected_data = data_flat(idx);

end

