function [ all_data, number_fingers ] = process_mat_file( fileID )
%process_mat_file Given a ".mat" file loads and processes

% Load file
load(fileID);

% Delete fileID Name
clear fileID;

% Grab variable
variables = who('');

if (isequal(variables{1}, 'variables'))
    all_data = eval(cell2mat(variables(2)));
else
    all_data = eval(cell2mat(variables(1)));
end

if (isfield(all_data, 'biotac'))
    number_fingers = size(all_data.biotac,2);
else
    number_fingers = size(all_data,2);
end


end

