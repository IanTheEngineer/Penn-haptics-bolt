function [loaded_data, total_files] = load_all_data(path, mat, segment)
% This function is similar to load_data_from_file but instead of a single
% file, it will load a directory of files, including other directories.  It
% expects the data to be stored in a heirachal fashion.  You must pass in
% the path of the directory.

% So for example I wanted to load some poke data.  My data would be stored in a dir called
% poke with subdirectories named 'plushie' and 'book'.  The actual
% data files should not contain any additional numbers other than the
% example number.  So data -> plushie -> poke01.txt, poke02.txt,
% stroke01.txt and data -> book -> poke01.txt, poke02.txt, stroke01.txt

% This will load the data into a struct where on the highest level are the
% different objects (plushie, book) and the next level will contain all of
% the motion (file names).

% To pull specific motions from the structure see select_motion.m

% Supports both JSON and txt files.  Note: careful naming conventions of
% files names are essential

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

% Checks if this is being run on linux or Windows
if ispc()
    dirLink = '\';
else
    dirLink = '/';
end

% File storage variables
data_file_names = dirr(path);
data_files = struct();
total_files = 1;

if (mat)
    for i = 1:numel(data_file_names)
        data = load_data_from_path(strcat(path, dirLink,data_file_names(i).name));
        total_files = total_files +1;
        if (segment)
            data_files(i).values = separate_states(data, true);
        else
            data_files(i).values = data;
        end
        data_files(i).values.name = data_file_names(i).name;
    end
else
    for i = 1:numel(data_file_names)
        data_files(i).name = data_file_names(i).name;
        data = struct();
        for j = 1:numel(data_file_names(i).isdir)
            fileID = fopen(strcat(path,dirLink,data_file_names(i).name,dirLink,data_file_names(i).isdir(j).name));

            % Populate information about the file
            name = regexp(data_file_names(i).isdir(j).name,'[0-9]','split');
            data(j).motion = name(1);
            data(j).name = data_file_names(i).name;
            data(j).category = i;

            % Load the data depending on .json or .txt format
            if (sum(strfind(data_file_names(i).isdir(j).name, '.json')) > 0)
                [all_data, number_fingers] = process_json_file(fileID);
            else
                [all_data, number_fingers] = process_txt_file(fileID);
            end

            data(j).values = all_data;
            total_files = total_files + 1;
        end
        data_files(i).data = data;
    end
end

loaded_data = data_files;