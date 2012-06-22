% Main script for launching the BioTac Replay GUI.
% See http://bolt-haptics.seas.upenn.edu/index.php/Software/MatlabGUI for
% a tutorial on how to use the GUI
% The sensor data that is being displayed is Syntouch BioTacs: 
% http://www.syntouchllc.com/Technology/TechnologyOverview.php

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

% Close previous GUI (will close all other figures open as well)
gui_handle = findall(0,'Type','figure');
delete(gui_handle);

% Clears previous matlab stored values (Comment out of neccessary)
clc
clear all;

% Add to path utilities folder
if ispc()
    addpath('..\utilities');
    javaaddpath('..\utilities\json-smart-1.1.1.jar');
    
else
    addpath('../utilities');
    javaaddpath('../utilities/json-smart-1.1.1.jar');
end

% Open up biotac GUI
BioTact_Replay;

