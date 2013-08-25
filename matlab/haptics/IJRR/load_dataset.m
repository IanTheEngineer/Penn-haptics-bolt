function [ loaded_data ] = load_dataset( runs )
%load_dataset Function that allows you to load objects into a struct form
% and contains the object runs that are specified.  This assumes you are
% pulling the same object runs from all of the objects.  If the runs are to
% be randomized, please change line 18 below to what you need it to do
%
% The user has to specify what runs they would like to load in 
% the form of an array 
%
% Ex. load_object([1 2 3]) will load the first three runs of the
% object selected
%
% If you would like to load only the human adjective data, you
% can call the function with an empty array - load_object([]) and
% only the human data will be loaded
%
% If you do not wish to load all of the robot data and only specific files,
% then create a file called "data_load_types.txt".  On one line - separated
% by commas, put the filenames of the files you wish to load.  This file
% must be located in the same folder as the scripts.  The script will only
% look at the first line of the file!
%
% Ex. biotac_pub.csv,simple_gripper_controller_state.csv
% on the first line will only load these two files for all objects selected

% Ask user to select objects
objects_path = uigetdir2();
loaded_data = -1;

% Check if user pressed cancel
if isequal(objects_path,0)
   disp('User pressed cancel')
   return;
end

loaded_data = struct();
for i = 1:numel(objects_path)
    
    % Get the object path and name
    folder_path = objects_path{i};
    object_name = regexp(folder_path,filesep,'split');
    object_name = object_name{end};
    
    % load the specific object
    data = load_object(folder_path,runs);

    % store the data
    eval(strcat('loaded_data.',object_name,' = data;'));
    
end


end

