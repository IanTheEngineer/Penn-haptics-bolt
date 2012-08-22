%% features from gripper aperture

% Using the joint_position information from gripper aperture

% 1) Calculate the area under the curves for each of the 5 phased as well as
% the total area

% 2) Find the peak value of TAP (which should reflect the width of
% objects) and peak value of SQUEEZE, and get the difference between the
% two which is called "sqz_depth" that should be related to adjectives like
% soft, deformable, squisky, elastic, etc.





% Load data from .h5 files
clear all_data
all_data = process_h5_file('cork_board_501.h5',0);

% Check the controller state to find the start moment of each segmentation 
tap_start=zeros(1,10);
sqz_start=zeros(1,10);
sqz_stop=zeros(1,10);
hold_start=zeros(1,10);
hold_stop=zeros(1,10);
slow_slide_start=zeros(1,10);
slow_slide_stop=zeros(1,10);
fast_slide_start=zeros(1,10);
fast_slide_stop=zeros(1,10);
done=zeros(1,10);

%  TAP = 5
%  THERMAL_HOLD = 1
%  SLIDE = 2
%  SQUEEZE = 3
%  SLIDE_FAST = 6
%  DONE = 4


for trial=1:10
    
    hold_start(trial)=find(all_data(trial).data.controller_state.controller_state == 1,1,'first');
    
    tap_start(trial)=find(all_data(trial).data.controller_state.controller_state == 5,1,'first');

    sqz_start(trial)=find(all_data(trial).data.controller_state.controller_state == 3,1,'first');

    slow_slide_start(trial)=find(all_data(trial).data.controller_state.controller_state == 2,1,'first');
    
    fast_slide_start(trial)=find(all_data(trial).data.controller_state.controller_state == 6,1,'first');

    done(trial)=find(all_data(trial).data.controller_state.controller_state == 4,1,'first');
    
     
end


%% Calculate the area of the 

steplength=0.01; % sampling rate is 100Hz

area_tap=zeros(1,10);
area_squeeze=zeros(1,10);
area_hold=zeros(1,10);
area_slow_slide=zeros(1,10);
area_fast_slide=zeros(1,10);

area_total=zeros(1,10);


for trial=1:10
    
    % Normalize the data 
    all_data(trial).data.gripper_aperture.joint_position = all_data(trial).data.gripper_aperture.joint_position - mean(all_data(trial).data.gripper_aperture.joint_position(1:15));
    

    % Area for taping
    clear tac_int

    for count=1:( sqz_start(trial)-tap_start(trial) )
        if count>1
            tac_int(count)=tac_int(count-1)+ ( all_data(trial).data.gripper_aperture.joint_position(count+tap_start(trial)-2)+all_data(trial).data.gripper_aperture.joint_position(count+tap_start(trial)-1) )/2*steplength;
        else
            tac_int(count)=all_data(trial).data.gripper_aperture.joint_position(tap_start(trial));
        end
    end
    area_tap(trial) = tac_int(count);
    
    
    % Area for squeezing
    clear tac_int

    for count=1:( hold_start(trial)-sqz_start(trial) )
        if count>1
            tac_int(count)=tac_int(count-1)+ ( all_data(trial).data.gripper_aperture.joint_position(count+sqz_start(trial)-2)+all_data(trial).data.gripper_aperture.joint_position(count+sqz_start(trial)-1) )/2*steplength;
        else
            tac_int(count)=all_data(trial).data.gripper_aperture.joint_position(sqz_start(trial));
        end
    end
    area_squeeze(trial) = tac_int(count);

    
    % Area for holding
    clear tac_int

    for count=1:( slow_slide_start(trial)-hold_start(trial) )
        if count>1
            tac_int(count)=tac_int(count-1)+ ( all_data(trial).data.gripper_aperture.joint_position(count+hold_start(trial)-2)+all_data(trial).data.gripper_aperture.joint_position(count+hold_start(trial)-1) )/2*steplength;
        else
            tac_int(count)=all_data(trial).data.gripper_aperture.joint_position(hold_start(trial));
        end
    end
    area_hold(trial) = tac_int(count);

    
    % Area for slow slide
    clear tac_int

    for count=1:( fast_slide_start(trial)-slow_slide_start(trial) )
        if count>1
            tac_int(count)=tac_int(count-1)+ ( all_data(trial).data.gripper_aperture.joint_position(count+slow_slide_start(trial)-2)+all_data(trial).data.gripper_aperture.joint_position(count+slow_slide_start(trial)-1) )/2*steplength;
        else
            tac_int(count)=all_data(trial).data.gripper_aperture.joint_position(slow_slide_start(trial));
        end
    end
    area_slow_slide(trial) = tac_int(count);
    
        
    % Area for fast slide
    clear tac_int

    for count=1:( done(trial)-fast_slide_start(trial) )
        if count>1
            tac_int(count)=tac_int(count-1)+ ( all_data(trial).data.gripper_aperture.joint_position(count+fast_slide_start(trial)-2)+all_data(trial).data.gripper_aperture.joint_position(count+fast_slide_start(trial)-1) )/2*steplength;
        else
            tac_int(count)=all_data(trial).data.gripper_aperture.joint_position(fast_slide_start(trial));
        end
    end
    area_fast_slide(trial) = tac_int(count);
    
    
    % Total area
    area_total(trial) = area_tap(trial)+area_squeeze(trial)+area_hold(trial)...
        + area_slow_slide(trial)+area_fast_slide(trial);

end



%% Find the max values of TAP, SQUEEZE, and the difference between them

peak_tap=zeros(1,10);
peak_squeeze=zeros(1,10);

sqz_depth=zeros(1,10);


for trial=1:10
    
    
    % Calculate the "softness" of the object
    peak_tap(trial)=min(all_data(trial).data.gripper_aperture.joint_position(tap_start(trial):sqz_start(trial)));
    peak_squeeze(trial)=min(all_data(trial).data.gripper_aperture.joint_position(sqz_start(trial):hold_start(trial)));

    
    sqz_depth(trial)=peak_tap(trial)-peak_squeeze(trial);
    

end

sqz_depth_mean = mean (sqz_depth);


