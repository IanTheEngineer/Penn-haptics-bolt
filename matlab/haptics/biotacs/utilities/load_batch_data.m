function [] = load_batch_data(path)
% batch load the files

% Checks if this is being run on linux or Windows
if ispc()
    dirLink = '\';
else
    dirLink = '/';
end

% File storage variables
data_file_names = dirr(path);

% For each file load and save
for i = 1:size(data_file_names, 1)
    name = regexp(data_file_names(i).name,'\.','split');
    %pathname = regexp(path, strcat('\',dirLink),'split');
    %save_file_name = strcat(pathname{1}, '_', name{1});
    save_file_name = name{1};
    varname = genvarname(save_file_name);
    eval([varname '= load_data_from_path(strcat(path,dirLink,data_file_names(i).name))']);
   
    save(name{1}, varname);
end

    

