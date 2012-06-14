function [ played_data ] = play_BioTact_Data()
%play_BioTact_Data Play BioTac Data using GUI in BioTact_Replay.m
% Outputs the bioTact information

% This is the main function that is executed when "Play" is clicked on the
% GUI.  Will cycle through the loaded data and plot each point one by one
% to simulate "real time".
%
% all_vals_graph: channels that will be displayed on large graph
% val1_graph: single channel to be displayed on first graph
% val2_graphs: single channel to be displayed on second graph
% start_time: time for when to start data playback
% end_time: time for when to end data playback
% filename: name of file to load
% all_data: converted data
% jButtonArray: array of buttons (19-electrodes) used to change color
% electrode_points: stored values for location in mm of the electrodes on
% the finger.  Used for calculating area of contact on the BioTac Sensor
% channel_display: The finger that was selected to display

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

% Variables to store global values for the simple large graph
global all_vals_graph val1_graph val2_graph start_time end_time filename
global jButtonArray all_data electrode_points channel_display 

% Data gain variables - current just an estimate - 3 Newtons
x_gain = 3;
y_gain = 3;
z_gain = 3;

% Data - normal force component in each direction - values given from Syntouch
n_x = [0 0 0 0 0 0 0.341853849 0.341853849 0.341853849 0.341853849 0 0 0 0 0 0 0 0 0];
n_y = [0.972680399 0.686249342 0.686249342 0.972680399 0.686249342 0.972680399 0 0 0 ...
    0 -0.972680399 -0.686249342 -0.686249342 -0.972680399 -0.686249342 -0.972680399 0 0 0];
n_z = [0.232148317 0.727366373 0.727366373 0.232148317 0.727366373 0.232148317 ...
    0.93975313 0.93975313 0.93975313 0.93975313 0.232148317 0.727366373 0.727366373 ...
    0.232148317 0.727366373 0.232148317 1 1 1];
 
% Get GUI Handles
handles = guidata(BioTact_Replay);

% Display what file has been loaded
if (isempty(filename))
    display('Please Load a File');
else
    set(handles.data_file,'string',num2str(filename));
end

% Check Time
if (isempty(start_time))
    start_time = 1;
end
if (isempty(end_time))
    end_time = size(all_data(1).pdc,1);
end

% Displays what channels that are playing
set(handles.channels_displayed,'string',num2str(all_vals_graph));

% Displays the times
set(handles.start_time,'string',num2str(start_time));
set(handles.end_time,'string',num2str(end_time));

% For triaxial force
% h = figure;

% Plot
hold(handles.All_Values,'on');
hold(handles.Select_Val1,'on');
hold(handles.Select_Val2,'on');
allValues = [all_data(channel_display).electrodes(:); all_data(channel_display).pdc(:)];
axis(handles.All_Values,[start_time end_time min(min(all_data(channel_display).electrodes(:)),-100) max(max(all_data(channel_display).electrodes(:)),100)]);
axis(handles.Select_Val1,[start_time end_time min(min(allValues), -100) max(max(allValues),100)]);
axis(handles.Select_Val2,[start_time end_time min(min(allValues),-100) max(max(allValues),100)]);

% If AC pressure is selected, resize the scale of the graphs
if (val1_graph == 23)
    allValues = [all_data(channel_display).electrodes(:); all_data(channel_display).pdc(:); all_data(channel_display).pac(:)];
    axis(handles.Select_Val1,[start_time end_time*22 min(min(allValues),-100) max(max(allValues),100)]);
elseif (val1_graph > 19)
    allValues = [all_data(channel_display).tac(:); all_data(channel_display).pdc(:); all_data(channel_display).tdc(:)];
    axis(handles.Select_Val1,[start_time end_time min(min(allValues),-100) max(max(allValues),100)]);
end
if (val2_graph == 23)
    allValues = [all_data(channel_display).electrodes(:); all_data(channel_display).pdc(:); all_data(channel_display).pac(:)];
    axis(handles.Select_Val2,[start_time end_time*22 min(min(allValues),-100) max(max(allValues),100)]);
elseif (val2_graph > 19)
    allValues = [all_data(channel_display).tac(:); all_data(channel_display).pdc(:); all_data(channel_display).tdc(:)];
    axis(handles.Select_Val2,[start_time end_time min(min(allValues),-100) max(max(allValues),100)]);
end

% Size of plot
data_length = size(all_data(channel_display).pdc,1);
% store area information
area_contact = zeros(1,size(all_data(1).pdc,1));
tic
% Checks if there are no channels selected to plot
if (isempty(all_vals_graph) && isempty(val1_graph) && isempty(val2_graph)) % Checks if there is nothing to graph
    % DO NOTHING :)
else
    for stepT = start_time:end_time
        
        % This is currently the time-step for each reading
        pause(0.001);
          
        % Go through each channel and plot one point - with erasemode
        % 'none' the points don't need to be replotted each step
        % Plots to the large graph
        endTime = min(stepT+1,min(end_time,data_length));
        toPlot = all_data(1).electrodes(stepT:endTime,[all_vals_graph]);
        t = stepT:endTime;
        [maxElectrodeVal, maxElectrode] = max(toPlot(1,:));
        plot(handles.All_Values, t,toPlot, 'erasemode','none');
        
        % Display the max electrode on GUI
        set(handles.point_of_force,'string',num2str(maxElectrode));

        % Compute force in each direction (x,y,z) - using default gain
        % Method from Syntouch
        f_x_sum = sum(n_x.*toPlot(1,:));
        f_y_sum = sum(n_y.*toPlot(1,:));
        f_z_sum = sum(n_z.*toPlot(1,:));
        
        f_x = x_gain*f_x_sum;
        f_y = y_gain*f_y_sum;
        f_z = z_gain*f_z_sum;
        
        % Display the max electrode on GUI
        set(handles.f_x_box,'string',num2str(round(f_x)));
        set(handles.f_y_box,'string',num2str(round(f_y)));
        set(handles.f_z_box,'string',num2str(round(f_z)));
        
        % Calculate area of electrodes
        thresh_points = toPlot(1,:) > 100;
        if sum(thresh_points) > 2
            area_points = electrode_points(thresh_points,:);

            dt = DelaunayTri(area_points(:,1),area_points(:,2));
    %         triplot(dt);
    %         axis equal;
            [ch areaC] = convexHull(dt);
            set(handles.area_contact,'string',num2str(round(areaC)));
            area_contact(stepT) = areaC;
        else
            set(handles.area_contact,'string',num2str(round(0)));
        end


        % Plot first small graph
        if (~isempty(val1_graph))
            if (val1_graph == 23)
                toPlot = all_data(channel_display).pac(stepT,:);
                t = ((stepT-1)*22+1:stepT*22);
            else
                t = stepT:endTime;
                endTime = min(stepT+1,min(end_time,data_length));
                if (val1_graph < 20)
                    toPlot = all_data(channel_display).electrodes(stepT:endTime,val1_graph);
                else
                    switch (val1_graph)
                        case 20
                            toPlot = all_data(channel_display).pdc(stepT:endTime);
                        case 21
                            toPlot = all_data(channel_display).tac(stepT:endTime);
                        case 22
                            toPlot = all_data(channel_display).tdc(stepT:endTime);
                    end
                end
            end
            plot(handles.Select_Val1, t,toPlot,'erasemode','none');
        end
        
        % Plot second small graph
        if (~isempty(val2_graph))
            if (val2_graph == 23)
                toPlot = all_data(channel_display).pac(stepT,:);
                t = ((stepT-1)*22+1:stepT*22);
            else
                t = stepT:endTime;
                endTime = min(stepT+1,min(end_time,data_length));
                if (val2_graph < 20)
                    %toPlot = 6.083./(all_data(1).electrodes(stepT:endTime,val2_graph)+1.976);
                    toPlot = all_data(channel_display).electrodes(stepT:endTime,val2_graph);
                else
                    switch (val2_graph)
                        case 20
                            toPlot = all_data(channel_display).pdc(stepT:endTime);
                        case 21
                            toPlot = all_data(channel_display).tac(stepT:endTime);
                        case 22
                            toPlot = all_data(channel_display).tdc(stepT:endTime);
                    end
                end
            end
            plot(handles.Select_Val2, t,toPlot,'erasemode','none');
        end
                
        % Change button colors depending on values
        for button = 1:19
            if (mod(stepT,1) == 0)
                rgb_val_all = all_data(channel_display).color(stepT,:,:);
                rgb_val = rgb_val_all(:,button,:);
                rgb_val = [rgb_val(:)]';
                jButtonArray{button}.set('background',java.awt.Color(rgb_val(1),rgb_val(2),rgb_val(3)));
            end
        end
        
    end
    hold(handles.All_Values,'off');
    hold(handles.Select_Val1,'off');
    hold(handles.Select_Val2,'off');
end
toc


played_data = all_data;

% Uncomment to see area information

% figure(1)
% plot(area_contact);
% title('Area over Samples');
% xlabel('Samples (100 Hz)');
% ylabel('Area (mm^2)');
% 
% figure(2)
% hist(fft(all_data(1).pac,129));
end

