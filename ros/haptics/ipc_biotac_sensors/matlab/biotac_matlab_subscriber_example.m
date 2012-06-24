% IPC-Bridge BioTacHand Example Subscriber
% 6/18/2012
% Ian McMahon
clear all;
clc;

% add the ipc_bridge_matlab binaries to your path
[a, p] = system('rospack find ipc_biotac_sensors');
addpath(strcat(p, '/bin')); 

% create a subscriber that reads a biotac_sensors/BioTacHand message
pid=biotac_sensors_BioTacHand('connect','subscriber','biotac_subscriber_module','biotachand');

% create an empty biotac_sensors/BioTacHand message structure
msg=biotac_sensors_BioTacHand('empty');

% read a message and print to screen
while (1)
    msg = biotac_sensors_BioTacHand('read',pid,1000)
    
%     pause(1)
    
end
