function [loaded_data] = load_data_from_path(filename)
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

% Add to path JSON library - only for java JSON - not needed anymore
% if ispc()
%     javaaddpath(strcat(pwd,'\json-smart-1.1.1.jar')); 
% else
%     javaaddpath(strcat(pwd,'/json-smart-1.1.1.jar'));
% end

% Opens the file
if (sum(strfind(filename, '.mat')))
    fileID = filename;
else
    fileID = fopen(filename);
end

% Process data
if (sum(strfind(filename, '.json')) > 0)
    [all_data, number_fingers] = process_json_file(fileID);
elseif (sum(strfind(filename, '.txt')) > 0)
    [all_data, number_fingers] = process_txt_file(fileID);
else
    [all_data, number_fingers] = process_mat_file(fileID);
end

loaded_data = all_data;