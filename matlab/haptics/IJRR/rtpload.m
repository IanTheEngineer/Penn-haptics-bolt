function  [time, data] = rtpload(filename)
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
fclose(fid);

% Make sure the file contains something.
if (line <0)
  error('Empty file %s', filename);
end

% Load the actual data.
% raw = load(filename);
raw = importdata(filename);

% Restructure the data.
column = 0;
while (~isempty(line))
  [token,line] = strtok(line,'%,');
  column = column+1;
  eval([token ' = raw.textdata(:,' num2str(column) ');']);
end

% Move to the correct output variables.  The first column
% is 'time', the rest are 'field.item1' 'field.item2' etc.
time = time;
data = field;

return;