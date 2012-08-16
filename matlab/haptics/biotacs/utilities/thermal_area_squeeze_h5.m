fileID = fopen('Aug15_area_squeeze.txt','a');

%% Load data from .h5 files
clear all_data
all_data = process_h5_file('cork_board_501.h5');

fprintf(fileID,'%s\n',all_data(1).name);

% Check the controller state to find the start moment of squeezing 
sqz_moment=zeros(1,10);

for trial=1:10
    for sample_number=1000:6000
        if all_data(trial).data.controller_state.controller_state(sample_number) == 3
            sqz_moment(trial)=sample_number;
            break
        end
    end
end


%% Integral of TAC IN BITS (area above TAC)

clf


for trial=1:10
    
    % Find the negative peak of TAC upong SQUEEZING
    clear tac_int area_above_tac

    valley_value = [all_data(trial).data.biotac(1).tac(1),all_data(trial).data.biotac(2).tac(1)];
    valley_point = [1,1];

    for column=1:2
        for count=(sqz_moment(trial)): (sqz_moment(trial)+1500)
            if all_data(trial).data.biotac(column).tac(count) < valley_value(column)
                valley_value(column)=all_data(trial).data.biotac(column).tac(count);
                valley_point(column)=count;
            end
        end
    end

    clear count

    % The time length is 3s that covers the TAC valley point 
    start_point(1) = valley_point(1) - 200;
    end_point(1) = valley_point(1) + 99; 

    start_point(2) = valley_point(2) - 200;
    end_point(2) = valley_point(2) + 99 ;

    % Check whether the TAC is under zero upon squeezing
    plot(all_data(trial).data.biotac(1).tac(start_point(1):end_point(1)))
    hold on

    
   % 
    
    % Calculate the area above TAC
    steplength = 1/100;  % sampling rate is 100 Hz

    tac_int=zeros(2,300);

    for raw=1:2
        for count=1:300
            if count>1
                tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+start_point(raw)-2)+all_data(trial).data.biotac(raw).tac(count+start_point(raw)-1) )/2*steplength;
            else
                tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(start_point(raw));
            end
        end
    end

    area_above_tac = [tac_int(1,count), tac_int(2,count)];

    fprintf(fileID,'%f %3f    ',area_above_tac);

    
    Trial = num2str(trial);

    fprintf(fileID,'Trial %s\n', Trial);
end


