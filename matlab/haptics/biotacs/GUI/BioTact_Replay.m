function varargout = BioTact_Replay(varargin)
% BIOTACT_REPLAY M-file for BioTact_Replay.fig
%      BIOTACT_REPLAY, by itself, creates a new BIOTACT_REPLAY or raises the existing
%      singleton*.
%
%      H = BIOTACT_REPLAY returns the handle to a new BIOTACT_REPLAY or the handle to
%      the existing singleton*.
%
%      BIOTACT_REPLAY('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BIOTACT_REPLAY.M with the given input arguments.
%
%      BIOTACT_REPLAY('Property','Value',...) creates a new BIOTACT_REPLAY or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BioTact_Replay_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BioTact_Replay_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BioTact_Replay_OpeningFcn, ...
                   'gui_OutputFcn',  @BioTact_Replay_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% End initialization code - DO NOT EDIT

% --- Executes just before BioTact_Replay is made visible.
function BioTact_Replay_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BioTact_Replay (see VARARGIN)

% Choose default command line output for BioTact_Replay
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BioTact_Replay wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = BioTact_Replay_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global jButtonArray electrode_points;
% Get default command line output from handles structure
varargout{1} = handles.output;

% Store handles to BioTac GUI
buttonNames = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','X1','X2','X3','X4'};
jButtonArray = getJButtonHandles(buttonNames);

% Store locations of the electrodes
electrode_points = [13.84 16.23;          % 1
          3.46*3 (5.41*2)+1; % 2
          3.46*3   8.26;  % 3
          13.84 5.41;          % 4
          3.46*3  1.944;      % 5
          13.84     0;          % 6
          6.92 20;               % 7
          6.92+3.33 20.5-3.33;              % 8
          6.92-3.33 20.5-3.33;              % 9
          6.92 20.5-6.66;              % 10
          0 16.23;              % 11
          3.46 (5.41*2)+1;  % 12
          3.46 8.26;          % 13
          0 5.41;              % 14
          3.46 1.944;         % 15
          0 0;                  % 16
          6.92 (5.41*2)+1;    % 17
          6.92 5.41;           % 18
          6.92 0];               % 19



% BEGIN PERSONALIZATION EDITS %

% --- Executes on button press in play.
function play_Callback(hObject, eventdata, handles)
% hObject    handle to play (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% clear all graphs
cla(handles.All_Values);
cla(handles.Select_Val1);
cla(handles.Select_Val2);
guidata(hObject, handles);

% Clear displayed Channels
set(handles.channels_displayed,'string',num2str(''));


clc
play_BioTact_Data();
guidata(hObject, handles);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% File Browsing/Data Input                                                %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% --- Executes on button press in browse_file_btn.
% This function asks user to find the data file they wish to collect and
% processes the data into a struct to be played back.  It loads the entire
% file
function browse_file_btn_Callback(hObject, eventdata, handles)
% hObject    handle to browse_file_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    global filename  total_channels all_data channel_display pr2_transforms start_time
    global end_time
    channel_display = 1; % Defaults to one finger first
    total_channels = 23;

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
    
    start_time = [];
    end_time = [];
    
    % Set what channels available
    set(handles.show_fingers,'string',num2str(number_fingers));
    
    % Set default channel
    set(handles.channel_select_display,'string',num2str(channel_display));
    
    % Set which file name is selected
    set(handles.data_file,'string',num2str(filename));


% --- Callback for the text box that shows tne file name
function data_file_Callback(hObject, eventdata, handles)
% hObject    handle to data_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% For displaying and selecting active finger channel                      %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% --- Displays the number of fingers available
function show_fingers_Callback(hObject, eventdata, handles)
% hObject    handle to show_fingers (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Callback that sets the finger that will be displayed for all the graphs
function channel_select_display_Callback(hObject, eventdata, handles)
% hObject    handle to channel_select_display (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global channel_display
channel_display = str2double(get(hObject,'String'));

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Graph Time Selection Values                                             %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% --- Sets the start time for all the graphs (sets when user enters values)
function start_time_Callback(hObject, eventdata, handles)
% hObject    handle to start_time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global start_time
start_time = str2double(get(hObject,'String'));

% --- Sets the end time for all the data (Sets when user enters values)
function end_time_Callback(hObject, eventdata, handles)
% hObject    handle to end_time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global end_time
end_time = str2double(get(hObject,'String'));


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Graph Channel Selection Functions                                       %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% --- Executes on selection change in Select_Val2_Menu.
% Select which channel to graph in Graph 1
function Select_Val1_Menu_Callback(hObject, eventdata, handles)
% hObject    handle to Select_Val2_Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
global val1_graph
graph_channel = contents{get(hObject,'Value')}; 
val1_graph = channel2num(graph_channel);

% --- Executes on selection change in Select_Val2_Menu.
% Select which channel to graph in Graph 2
function Select_Val2_Menu_Callback(hObject, eventdata, handles)
% hObject    handle to Select_Val2_Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
global val2_graph
graph_channel = contents{get(hObject,'Value')}; 
val2_graph = channel2num(graph_channel);

% --- Executes on selection change in Add_All_Val_Menu.
% Select which channel to graph in Graph All
function Add_All_Val_Menu_Callback(hObject, eventdata, handles)
% hObject    handle to Add_All_Val_Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global all_vals_store
contents = cellstr(get(hObject,'String'));
string_val = contents{get(hObject,'Value')};
all_vals_store = channel2num(string_val); 


% --- Executes on button press in add_channel_btn.
function add_channel_btn_Callback(hObject, eventdata, handles)
% hObject    handle to add_channel_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global all_vals_graph all_vals_store
all_vals_graph = [all_vals_graph all_vals_store];
all_vals_graph = unique(all_vals_graph); % Sort values
set(handles.channels_displayed,'string',num2str(all_vals_graph));
set(handles.Add_All_Val_Menu,'Value',1)

% --- Executes on button press in remove_channel_btn.
function remove_channel_btn_Callback(hObject, eventdata, handles)
% hObject    handle to remove_channel_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global all_vals_graph all_vals_store
all_vals_graph(all_vals_graph == all_vals_store) = [];
all_vals_graph = unique(all_vals_graph); % Sort values
set(handles.channels_displayed,'string',num2str(all_vals_graph));
set(handles.Add_All_Val_Menu,'Value',1)

% --- Executes on button press in clear_button.
% Clears all of the graphed channels for the large graph
function clear_button_Callback(hObject, eventdata, handles)
% hObject    handle to clear_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global all_vals_graph all_vals_store
all_vals_graph = [];
all_vals_store = [];
set(handles.channels_displayed,'string',num2str(all_vals_graph));
set(handles.Add_All_Val_Menu,'Value',1)


% --- Executes on button press in add_all_channels_btn.
% Adds all the graph channels to the large graph
function add_all_channels_btn_Callback(hObject, eventdata, handles)
% hObject    handle to add_all_channels_btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global all_vals_graph all_vals_store
all_vals_graph = [1:19];
all_vals_store = [];
set(handles.channels_displayed,'string',num2str(all_vals_graph));
set(handles.Add_All_Val_Menu,'Value',1)

% Displays the selected channels
function channels_displayed_Callback(hObject, eventdata, handles)
% hObject    handle to channels_displayed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Force Display                                                           %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Just displays the electrode where the maximum force is coming from
function point_of_force_Callback(hObject, eventdata, handles)
% hObject    handle to point_of_force (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of point_of_force as text
%        str2double(get(hObject,'String')) returns contents of point_of_force as a double

function f_x_box_Callback(hObject, eventdata, handles)
% hObject    handle to f_x_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f_x_box as text
%        str2double(get(hObject,'String')) returns contents of f_x_box as a double

function f_y_box_Callback(hObject, eventdata, handles)
% hObject    handle to f_y_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f_y_box as text
%        str2double(get(hObject,'String')) returns contents of f_y_box as a double


function f_z_box_Callback(hObject, eventdata, handles)
% hObject    handle to f_z_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of f_z_box as text
%        str2double(get(hObject,'String')) returns contents of f_z_box as a double


function area_contact_Callback(hObject, eventdata, handles)
% hObject    handle to area_contact (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of area_contact as text
%        str2double(get(hObject,'String')) returns contents of area_contact as a double


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Function Creators - no need to touch                                    %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% --- Executes during object creation, after setting all properties.
% Don't need to touch
function Select_Val2_Menu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Select_Val2_Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
% Don't need to touch
function Select_Val1_Menu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Select_Val2_Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
% Don't need to touch
function Add_All_Val_Menu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Add_All_Val_Menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
% Don't need to touch
function end_time_CreateFcn(hObject, eventdata, handles)
% hObject    handle to end_time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
% Don't need to touch
function start_time_CreateFcn(hObject, eventdata, handles)
% hObject    handle to start_time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function data_file_CreateFcn(hObject, eventdata, handles)
% hObject    handle to data_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function channels_displayed_CreateFcn(hObject, eventdata, handles)
% hObject    handle to channels_displayed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function point_of_force_CreateFcn(hObject, eventdata, handles)
% hObject    handle to point_of_force (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Used to display forces on finger (not used)                             %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


% --- Executes on button press in display1.
% Does nothing currently
function display1_Callback(hObject, eventdata, handles)
% hObject    handle to display1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display17.
function display17_Callback(hObject, eventdata, handles)
% hObject    handle to display17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display2.
function display2_Callback(hObject, eventdata, handles)
% hObject    handle to display2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in displayX1.
function displayX1_Callback(hObject, eventdata, handles)
% hObject    handle to displayX1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display10.
function display10_Callback(hObject, eventdata, handles)
% hObject    handle to display10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display8.
function display8_Callback(hObject, eventdata, handles)
% hObject    handle to display8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display9.
function display9_Callback(hObject, eventdata, handles)
% hObject    handle to display9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display7.
function display7_Callback(hObject, eventdata, handles)
% hObject    handle to display7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display11.
function display11_Callback(hObject, eventdata, handles)
% hObject    handle to display11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in displayX3.
function displayX3_Callback(hObject, eventdata, handles)
% hObject    handle to displayX3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display12.
function display12_Callback(hObject, eventdata, handles)
% hObject    handle to display12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display13.
function display13_Callback(hObject, eventdata, handles)
% hObject    handle to display13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display3.
function display3_Callback(hObject, eventdata, handles)
% hObject    handle to display3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display18.
function display18_Callback(hObject, eventdata, handles)
% hObject    handle to display18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display5.
function display5_Callback(hObject, eventdata, handles)
% hObject    handle to display5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display4.
function display4_Callback(hObject, eventdata, handles)
% hObject    handle to display4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display15.
function display15_Callback(hObject, eventdata, handles)
% hObject    handle to display15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display14.
function display14_Callback(hObject, eventdata, handles)
% hObject    handle to display14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display19.
function display19_Callback(hObject, eventdata, handles)
% hObject    handle to display19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display16.
function display16_Callback(hObject, eventdata, handles)
% hObject    handle to display16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in display6.
function display6_Callback(hObject, eventdata, handles)
% hObject    handle to display6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in displayX4.
function displayX4_Callback(hObject, eventdata, handles)
% hObject    handle to displayX4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in displayX2.
function displayX2_Callback(hObject, eventdata, handles)
% hObject    handle to displayX2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes during object creation, after setting all properties.
function f_x_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f_x_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function f_y_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f_y_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function f_z_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to f_z_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function area_contact_CreateFcn(hObject, eventdata, handles)
% hObject    handle to area_contact (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function show_fingers_CreateFcn(hObject, eventdata, handles)
% hObject    handle to show_fingers (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function channel_select_display_CreateFcn(hObject, eventdata, handles)
% hObject    handle to channel_select_display (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
