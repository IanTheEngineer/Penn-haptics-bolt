function  [time, data] = fast_rtpload(filename)
%
%   [time, data] = rtpload(filename)
%
%   Load data from a ROS message file, created with
%   a 'rostopic echo -p topic > filename' command.
%
%   filename    name (including path) of data file
%   time        Nx1 vector of ROS times, when the
%                individual messages were received
%   data        structure of Nx1 vectors, corresponding
%                to the fields in the message data

% Note: rostopic saves a header line with information
% about the data.  The first column is the time the
% message was received, the rest are message fields.

% Get the header line - which includes the data format.
fid = fopen(filename);
if (fid < 0)
  error('Unable to open file %s', filename);
end
line = fgetl(fid);
data_line = fgetl(fid); 

% Test the input to find out the format
% Only differentiates between numbers and text
split_data = regexp(data_line,',','split');
t_format = '';
for col = 1:length(split_data)
    test_double = isstrprop(strrep(split_data{col},'.',''),'digit');
    if all(test_double)
        append = ' %f';
    else
        append = ' %s';
    end
    
    t_format = [t_format append];
end
t_format = t_format(2:end);
    
frewind(fid); fgetl(fid); % reset the pointer to line after header
raw = textscan(fid,t_format,'Delimiter',',');

% Restructure the data.
column = 0;
while (~isempty(line))
  [token,line] = strtok(line,'%,');
  column = column+1;
  eval([token ' = raw{:,' num2str(column) '};']);
end

% Move to the correct output variables.  The first column
% is 'time', the rest are 'field.item1' 'field.item2' etc.
time = time;
data = field;

return;