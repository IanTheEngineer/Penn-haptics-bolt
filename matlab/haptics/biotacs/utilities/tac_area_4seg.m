
%% Load data from .h5 files
clear all_data
all_data = process_h5_file('acrylic_211.h5',0);


% fprintf(fileID,'%s\n',all_data(1).name);

% Check the controller state to find the start moment of each segmentation 

sqz_start=zeros(1,10);
sqz_stop=zeros(1,10);
hold_start=zeros(1,10);
hold_stop=zeros(1,10);
slow_slide_start=zeros(1,10);
slow_slide_stop=zeros(1,10);
fast_slide_start=zeros(1,10);
fast_slide_stop=zeros(1,10);



for trial =1:10
   
    clear detail_states
    
    detail_states = cellstr(all_data(trial).data.controller_state.controller_state_detail');
    
    sqz_start(trial) = find(ismember(detail_states, 'SQUEEZE_SET_PRESSURE_SLOW')==1,1,'first');
    sqz_stop(trial) = find(ismember(detail_states, 'SQUEEZE_SET_PRESSURE_SLOW')==1,1,'last');
    
    hold_start(trial) = find(ismember(detail_states, 'HOLD_FOR_10_SECONDS')==1,1,'first');
    hold_stop(trial) = hold_start(trial)+1000;
   
    slow_slide_start(trial) = find(ismember(detail_states, 'SLIDE_5CM')==1,1,'first');
    slow_slide_stop(trial) = find(ismember(detail_states, 'SLIDE_5CM')==1,1,'last');

    fast_slide_start(trial) = find(ismember(detail_states, 'MOVE_DOWN_5CM')==1,1,'first');
    fast_slide_stop(trial) = find(ismember(detail_states, 'MOVE_DOWN_5CM')==1,1,'last');

end

%  THERMAL_HOLD = 1
%  SLIDE = 2
%  SQUEEZE = 3
%  SLIDE_FAST = 6
%  DONE = 4


% for trial=1:10
%     total_samples = length(all_data(trial).data.biotac(1).pdc);
% 
%     % find the squeeze moment
%     for sample_number=1:total_samples
%         if all_data(trial).data.controller_state.controller_state(sample_number) == 3
%             mark=sample_number;
%             break
%         end
%     end    
%     
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(8,sample_number) == '_'
%             sqz_start(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end
%     
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'O'
%             sqz_stop(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end   
%  
%  
%     % find the holding moment
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state(sample_number) == 1
%             mark=sample_number;
%             break
%         end
%     end
% 
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'H'
%             hold_start(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end
%     
%     hold_stop(trial)=hold_start(trial)+999;
%     
%     
%     % For slow slide
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state(sample_number) == 2
%             mark=sample_number;
%             break
%         end
%     end
% 
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(7,sample_number) == '5'
%             slow_slide_start(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end   
%     
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'O'
%             slow_slide_stop(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end
%     
%     
%     
%     % For fast slide
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state(sample_number) == 6
%             mark=sample_number;
%             break
%         end
%     end
%     
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'M'
%             fast_slide_start(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end
%  
%     for sample_number=mark:total_samples
%         if all_data(trial).data.controller_state.controller_state_detail(1,sample_number) == 'O'
%             fast_slide_stop(trial)=sample_number;
%             mark=sample_number;
%             break
%         end
%     end
%      
% end


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
    

    % Area for squeezing
    area_squeeze(1,trial)=trapz(all_data(trial).data.biotac(1).tac(sqz_start(trial):sqz_stop(trial)));
    area_squeeze(2,trial)=trapz(all_data(trial).data.biotac(2).tac(sqz_start(trial):sqz_stop(trial)));
   
    % Area for holding
    area_hold(1,trial)=trapz(all_data(trial).data.biotac(1).tac(hold_start(trial):hold_stop(trial)));
    area_hold(2,trial)=trapz(all_data(trial).data.biotac(2).tac(hold_start(trial):hold_stop(trial)));
   
    % Area for slow slide
    area_slow_slide(1,trial)=trapz(all_data(trial).data.biotac(1).tac(slow_slide_start(trial):slow_slide_stop(trial)));
    area_slow_slide(2,trial)=trapz(all_data(trial).data.biotac(2).tac(slow_slide_start(trial):slow_slide_stop(trial)));
    
    % Area for fast slide
    area_fast_slide(1,trial)=trapz(all_data(trial).data.biotac(1).tac(fast_slide_start(trial):fast_slide_stop(trial)));
    area_fast_slide(2,trial)=trapz(all_data(trial).data.biotac(2).tac(fast_slide_start(trial):fast_slide_stop(trial)));

    
%     % Area for squeezing
%     clear tac_int area_squeeze area_hold area_slow_slide area_fast_slide
% 
%     for raw=1:2
%         for count=1:(sqz_stop(trial)-sqz_start(trial)+1)
%             if count>1
%                 tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+sqz_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+sqz_start(trial)-1) )/2*steplength;
%             else
%                 tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(sqz_start(trial));
%             end
%         end
%     end
% 
%     area_squeeze = [tac_int(1,count), tac_int(2,count)];
%    
%     
%     % Area for holding
%     clear tac_int
% 
%     for raw=1:2
%         for count=1:1000
%             if count>1
%                 tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+hold_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+hold_start(trial)-1) )/2*steplength;
%             else
%                 tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(hold_start(trial));
%             end
%         end
%     end
% 
%     area_hold = [tac_int(1,count), tac_int(2,count)];
%   
%  
%     % Area for slow slide
%     clear tac_int
% 
%     for raw=1:2
%         for count=1:( slow_slide_stop(trial) - slow_slide_start(trial) +1 )
%             if count>1
%                 tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+slow_slide_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+slow_slide_start(trial)-1) )/2*steplength;
%             else
%                 tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(slow_slide_start(trial));
%             end
%         end
%     end
% 
%     area_slow_slide = [tac_int(1,count), tac_int(2,count)];
%     
%     
%     % Area for fast slide
%     clear tac_int
% 
%     for raw=1:2
%         for count=1:( fast_slide_stop(trial) - fast_slide_start(trial) +1)
%             if count>1
%                 tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+fast_slide_start(trial)-2)+all_data(trial).data.biotac(raw).tac(count+fast_slide_start(trial)-1) )/2*steplength;
%             else
%                 tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(fast_slide_start(trial));
%             end
%         end
%     end
% 
%     area_fast_slide = [tac_int(1,count), tac_int(2,count)];
% 
    
%   fprintf(fileID,'%f    ',area_squeeze);
%   fprintf(fileID,'%f    ',area_hold);
%   fprintf(fileID,'%f    ',area_slow_slide);
%   fprintf(fileID,'%f    ',area_fast_slide);
% 
%   Trial = num2str(trial);
%     
%   fprintf(fileID,'Trial  %s\n', Trial);
  

end




