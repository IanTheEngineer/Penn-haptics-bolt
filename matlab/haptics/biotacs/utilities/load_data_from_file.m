function [loaded_data] = load_data_from_file()
% This function will ask the user for a single file to load.  It will parse
% the either .txt Syntouch output format or ROS .json message format

% Input: Syntouch .txt output file or ROS .json format from biotac_stack
% Output: structure of format: struct('finger',{},'electrodes',{},'tdc',{}, 'tac', {},'pac',{},'pdc',{});
% Multiple fingers are in an array of structs.

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012


[filename, pathname, filterindex] = uigetfile('*.json;*.txt', 'Pick a file');
    
% Check if user pressed cancel
if isequal(filename,0) || isequal(pathname,0)
   disp('User pressed cancel')
   return;
else
   disp(['User selected ', fullfile(pathname, filename)])
end

% Opens the file
fileID = fopen(strcat(pathname,filename));

if (sum(strfind(filename, '.json')) > 0)
    [all_data, number_fingers] = process_json_file(fileID);
else
    [all_data, number_fingers] = process_txt_file(fileID);
end

loaded_data = all_data;