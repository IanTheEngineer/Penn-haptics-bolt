function [ loaded_data ] = load_single_run( run_path )
%load_single_run Summary of this function goes here
%   Detailed explanation goes here

% We know all of the runs are stored in the 'csv' folder
file_path = fullfile(run_path,'csv');

% if there exist a file "data_load_types" - then only load
% those specific data types from the file, otherwise load all
% contained in the folder - data_load_types must be in the same
% folder as all of the scripts
if exist('data_load_types.txt','file')
    fid = fopen('data_load_types.txt');
    if (fid < 0)
      error('Unable to open file %s', filename);
    end
    line = fgetl(fid);
    fclose(fid);
    files = regexp(line,',','split');
else
    files = dir(file_path);
    files = {files(~[files.isdir]).name};
end
loaded_data = struct();
for i = 1:size(files,2)
    
    % Load each file and store in struct
    [~,data] = fast_rtpload(fullfile(file_path,files{i}));
    file_name = regexp(files{i},'\.','split');
    
    eval(strcat('loaded_data.',file_name{1},'= data;'));
end


end

