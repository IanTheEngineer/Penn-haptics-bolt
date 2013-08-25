function [loaded_data] = load_object(folder_path,runs)
% This function will ask the user for a single object to load.  
% The user has to specify what runs they would like to load in 
% the form of an array 
%
% Ex. load_object([1 2 3]) will load the first three runs of the
% object selected
%
% If you would like to load only the human adjective data, you
% can call the function with an empty array - load_object([]) and
% only the human data will be loaded

%folder_path = uigetdir2();
loaded_data = -1;
% Check if user pressed cancel
% if isequal(folder_path,0)
%    disp('User pressed cancel')
%    return;
% end

% Get the object name from the folder path
object_name = regexp(folder_path,filesep,'split');
object_name = object_name{end};
object_id = regexp(object_name,'_','split');
object_id = char(object_id{end});

% Make sure the user actually selected an object
if ~isstrprop(object_id,'digit')
    disp(['User selected "', object_name,'", which is not a valid object folder']);
    return;
else
   object_name = regexp(folder_path,filesep,'split');
   disp(['User selected ', object_name{end}])
end

% Load each of the runs into a struct form
loaded_data = struct();
for i = runs
    
    % data is stored by "runs" - so we know that each 
    % run is labeled "run_xx"
    run_name = strcat('run_',sprintf('%02g',i));
    run_path = fullfile(folder_path,run_name);

    %load all data for the run
    single_run = load_single_run(run_path);
    
    % Store the data
    eval(strcat('loaded_data.robot.',run_name,' = single_run;'));
end

% load adjective data
% get the adjective files
[~,loaded_data.humans.binary_ratings] = fast_rtpload(fullfile(folder_path,'adjective_labels','binary_ratings.csv'));
[~,loaded_data.humans.free_response] = fast_rtpload(fullfile(folder_path,'adjective_labels','free_response.csv'));
[~,loaded_data.humans.scale_ratings] = fast_rtpload(fullfile(folder_path,'adjective_labels','scale_ratings.csv'));













