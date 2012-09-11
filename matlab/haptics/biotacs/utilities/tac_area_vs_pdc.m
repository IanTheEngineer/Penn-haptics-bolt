fileID = fopen('Aug19_TACarea_vs_pdc.txt','a');


%% Load data from .h5 files
clear all_data
all_data = process_h5_file('cologne_container__602.h5');


fprintf(fileID,'%s\n',all_data(1).name);

% Check the controller state to find the start moment of each segmentation 
sqz_start=zeros(1,10);
sqz_stop=zeros(1,10);
hold_start=zeros(1,10);
hold_stop=zeros(1,10);
slow_slide_start=zeros(1,10);
slow_slide_stop=zeros(1,10);
fast_slide_start=zeros(1,10);
fast_slide_stop=zeros(1,10);


%  THERMAL_HOLD = 1
%  SLIDE = 2
%  SQUEEZE = 3
%  SLIDE_FAST = 6
%  DONE = 4


% For hold, slow slide, and fast slide, find the correlation between the
% area and the pressure

% pick the pressure of the moment 1sec after a motion start

for trial=1:10
    total_samples = length(all_data(trial).data.biotac(1).pdc);

    % For squeeze
    for sample_number=1:total_samples
        if all_data(trial).data.controller_state.controller_state(sample_number) == 3
            mark=sample_number;
            break
        end
    end    
    
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(8,sample_number) == '_'
            sqz_start(trial)=sample_number;
            mark=sample_number;
            break
        end
    end
    
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'O'
            sqz_stop(trial)=sample_number;
            mark=sample_number;
            break
        end
    end   
 
 
    % For hold
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state(sample_number) == 1
            mark=sample_number;
            break
        end
    end

    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'H'
            hold_start(trial)=sample_number;
            mark=sample_number;
            break
        end
    end
    
    hold_stop(trial)=hold_start(trial)+999;
    
    
    % For slow slide
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state(sample_number) == 2
            mark=sample_number;
            break
        end
    end

    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(7,sample_number) == '5'
            slow_slide_start(trial)=sample_number;
            mark=sample_number;
            break
        end
    end   
    
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'O'
            slow_slide_stop(trial)=sample_number;
            mark=sample_number;
            break
        end
    end
    
    
    
    % For fast slide
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state(sample_number) == 6
            mark=sample_number;
            break
        end
    end
    
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'M'
            fast_slide_start(trial)=sample_number;
            mark=sample_number;
            break
        end
    end
 
    for sample_number=mark:total_samples
        if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'O'
            fast_slide_stop(trial)=sample_number;
            mark=sample_number;
            break
        end
    end
     
end


%% Area above TAC for SQEEZE, HOLD, SLOW SLIDE, and FAST SLIDE

steplength = 1/100;  % sampling rate is 100 Hz

area_squeeze=zeros(2,10);
area_hold=zeros(2,10);
area_slow_slide=zeros(2,10);
area_fast_slide=zeros(2,10);

pressure_hold=zeros(2,10);
pressure_slow_slide=zeros(2,10);
pressure_fast_slide=zeros(2,10);


for trial=1:10
    
    % Normalize the TAC value by subtracting the part after the squeezing
    % begins    
    all_data(trial).data.biotac(1).tac = all_data(trial).data.biotac(1).tac - mean(all_data(trial).data.biotac(1).tac(sqz_start(trial):(sqz_start(trial)+50)));
    all_data(trial).data.biotac(2).tac = all_data(trial).data.biotac(2).tac - mean(all_data(trial).data.biotac(2).tac(sqz_start(trial):(sqz_start(trial)+50)));

    
    % Area for squeezing
    clear tac_int 

    for raw=1:2
        for count=1:(sqz_stop(trial)-sqz_start(trial)+1)
            if count>1
                tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+sqz_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+sqz_start(trial)-1) )/2*steplength;
            else
                tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(sqz_start(trial));
            end
        end
        area_squeeze(raw,trial) = tac_int(raw,count);   % here the raw is actually column
    end  
    
     
    
    % Area for holding
    clear tac_int

    for raw=1:2
        for count=1:1000
            if count>1
                tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+hold_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+hold_start(trial)-1) )/2*steplength;
            else
                tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(hold_start(trial));
            end
        end
        pressure_hold(raw,trial) = all_data(trial).data.biotac(raw).pdc(hold_start(trial)+100);
        area_hold(raw,trial) = tac_int(raw,count);
    end

  

    % Area for slow slide
    clear tac_int

    for raw=1:2
        for count=1:( slow_slide_stop(trial) - slow_slide_start(trial) +1 )
            if count>1
                tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+slow_slide_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+slow_slide_start(trial)-1) )/2*steplength;
            else
                tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(slow_slide_start(trial));
            end
        end
        pressure_slow_slide(raw,trial) = all_data(trial).data.biotac(raw).pdc(slow_slide_start(trial)+100);
        area_slow_slide(raw,trial) = tac_int(raw,count);
    end

      
    
    % Area for fast slide
    clear tac_int

    for raw=1:2
        for count=1:( fast_slide_stop(trial) - fast_slide_start(trial) +1)
            if count>1
                tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+fast_slide_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+fast_slide_start(trial)-1) )/2*steplength;
            else
                tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(fast_slide_start(trial));
            end
        end
        pressure_fast_slide(raw,trial) = all_data(trial).data.biotac(raw).pdc(fast_slide_start(trial)+100);
        area_fast_slide(raw,trial) = tac_int(raw,count);
    end

    
    
    % Plot the clusters
    subplot(2,1,1);
    plot(pressure_hold(1,trial),area_hold(1,trial),'*')
    hold on
    plot(pressure_slow_slide(1,trial),area_slow_slide(1,trial),'s')
    hold on
    plot(pressure_fast_slide(1,trial),area_fast_slide(1,trial),'+')
    hold on
    title('Finger 1')
    
    subplot(2,1,2);
    plot(pressure_hold(1,trial),area_hold(2,trial),'*')
    hold on
    plot(pressure_slow_slide(1,trial),area_slow_slide(2,trial),'s')
    hold on
    plot(pressure_fast_slide(1,trial),area_fast_slide(2,trial),'+')
    hold on
    title('Finger 2') 

    
%   fprintf(fileID,'%f    ',area_squeeze);
%   fprintf(fileID,'%f    ',area_hold);
%   fprintf(fileID,'%f    ',area_slow_slide);
%   fprintf(fileID,'%f    ',area_fast_slide);
% 
%   Trial = num2str(trial);
%     
%   fprintf(fileID,'Trial  %s\n', Trial);
%   

end


%% Linear fits for area above TAC and correspondent PDC

clear yfit_hold_1 yfit_hold_2 yfit_slow_slide_1 yfit_slow_slide_2 yfit_fast_slide_1 yfit_fast_slide_2

% HOLD = zeros(2,2);
% SLOW_SLIDE = zeros(2,2);
% FAST_SLIDE = zeros(2,2);


    HOLD_1 = polyfit(pressure_hold(1,:),area_hold(1,:),1);
    HOLD_2 = polyfit(pressure_hold(2,:),area_hold(2,:),1);
    yfit_hold_1 = HOLD_1(1)*pressure_hold(1,:) + HOLD_1(2);
    yfit_hold_2 = HOLD_2(1)*pressure_hold(2,:) + HOLD_2(2);

    SLOW_SLIDE_1 = polyfit(pressure_slow_slide(1,:),area_slow_slide(1,:),1);
    SLOW_SLIDE_2 = polyfit(pressure_slow_slide(2,:),area_slow_slide(2,:),1);
    yfit_slow_slide_1 = SLOW_SLIDE_1(1)*pressure_slow_slide(1,:) + SLOW_SLIDE_1(2);
    yfit_slow_slide_2 = SLOW_SLIDE_2(1)*pressure_slow_slide(2,:) + SLOW_SLIDE_2(2);
    
    
    FAST_SLIDE_1 = polyfit(pressure_fast_slide(1,:),area_fast_slide(1,:),1);
    FAST_SLIDE_2 = polyfit(pressure_fast_slide(2,:),area_fast_slide(2,:),1);
    yfit_fast_slide_1 = FAST_SLIDE_1(1)*pressure_fast_slide(1,:) + FAST_SLIDE_1(2);
    yfit_fast_slide_2 = FAST_SLIDE_2(1)*pressure_fast_slide(2,:) + FAST_SLIDE_2(2);
    
  
    subplot(1,3,1); 
    plot(pressure_hold(1,:),area_hold(1,:),'b*',pressure_hold(1,:),yfit_hold_1,'r-.');
    hold on
    plot(pressure_hold(2,:),area_hold(2,:),'bs',pressure_hold(2,:),yfit_hold_2,'r-.');
    title('Hold')
    legend('finger 1','','finger 2')
    
    subplot(1,3,2); 
    plot(pressure_slow_slide(1,:),area_slow_slide(1,:),'k*',pressure_slow_slide(1,:),yfit_slow_slide_1,'r-.');
    hold on
    plot(pressure_slow_slide(2,:),area_slow_slide(2,:),'ks',pressure_slow_slide(2,:),yfit_slow_slide_2,'r-.');
    title('Slow slide')
    legend('finger 1','','finger 2')
    
    subplot(1,3,3);
    plot(pressure_fast_slide(1,:),area_fast_slide(1,:),'m*',pressure_fast_slide(1,:),yfit_fast_slide_1,'r-.');
    hold on
    plot(pressure_fast_slide(2,:),area_fast_slide(2,:),'ms',pressure_fast_slide(2,:),yfit_fast_slide_2,'r-.');
    title('Fast slide')
    legend('finger 1','','finger 2')
    



fprintf(fileID,'HOLD %2f %3f \n', HOLD_1, HOLD_2);
fprintf(fileID,'SLOW SLIDE %2f %3f \n', SLOW_SLIDE_1,SLOW_SLIDE_2);
fprintf(fileID,'FAST SLIDE %2f %3f \n', FAST_SLIDE_1,FAST_SLIDE_2);




