fileID = fopen('Aug13_area.txt','a');

%% Load data from .h5 files
clear all_data

all_data = process_h5_file('chalkboard_eraser_408.h5');

fprintf(fileID,'%s\n',all_data(1).name);

% Check the controller state to find the start moment of holding 

hold_moment=zeros(1,10);

for trial=1:10
    for sample_number=3000:8000
        if all_data(trial).data.controller_state.controller_state(sample_number) == 1
            hold_moment(trial)=sample_number;
            break
        end
    end
end


%% Integral of TAC IN BITS (area above TAC)

% The integral begins with the negative peak of the TAC curve when holding

for trial=1:10
    
    % Find the negative peak of TAC
    clear tac_int area_above_tac

    valley_value = [all_data(trial).data.biotac(1).tac(1),all_data(trial).data.biotac(2).tac(1)];
    valley_point = [1,1];

    for column=1:2
        for count=(hold_moment(trial)+200): (hold_moment(trial)+1500)
            if all_data(trial).data.biotac(1).tac(count) < valley_value(column)
                valley_value(column)=all_data(trial).data.biotac(1).tac(count);
                valley_point(column)=count;
            end
        end
    end

    clear count

    % THe time length is 8s to make sure the fingers in contact with objects
    start_point(1) = valley_point(1);
    end_point(1) = valley_point(1) + 799; 

    start_point(2) = valley_point(2);
    end_point(2) = valley_point(2) + 799 ;


    % Calculate the area above TAC
    steplength = 1/100;  % sampling rate is 100 Hz

    tac_int=zeros(2,800);

    for raw=1:2
        for count=1:800
            if count>1
                tac_int(raw,count)=tac_int(raw,count-1)+ ( all_data(trial).data.biotac(raw).tac(count+start_point(raw)-2)+all_data(trial).data.biotac(raw).tac(count+start_point(raw)-1) )/2*steplength;
            else
                tac_int(raw,count)=all_data(trial).data.biotac(raw).tac(start_point(raw));
            end
        end
    end

    area_above_tac = [tac_int(1,count), tac_int(2,count)];

    fprintf(fileID,'%f %3f    ',area_above_tac);



%%========= Fit TDC and TAC into a linear curve to get TAU and T_final

    % Convert raw data into real temperature
    clf
    clear real_tac_1 real_tdc_1 real_tac_2 real_tdc_2

    num_frames_throw_away=50;

    for count=1:size(all_data(trial).data.biotac(1).tdc,1)
        real_tdc_1(count)=-273.15+4025/log( (155183-46555*all_data(trial).data.biotac(1).raw_tdc(count)/4095)/(all_data(trial).data.biotac(1).raw_tdc(count)/4095) );
        real_tac_1(count)=-41.07/log( (155183-46555*all_data(trial).data.biotac(1).raw_tac(count)/4095)/(all_data(trial).data.biotac(1).raw_tac(count)/4095) );
    end

    for count=1:size(all_data(trial).data.biotac(2).tdc,1)
        real_tdc_2(count)=-273.15+4025/log( (155183-46555*all_data(trial).data.biotac(2).raw_tdc(count)/4095)/(all_data(trial).data.biotac(2).raw_tdc(count)/4095) );
        real_tac_2(count)=-41.07/log( (155183-46555*all_data(trial).data.biotac(2).raw_tac(count)/4095)/(all_data(trial).data.biotac(2).raw_tac(count)/4095) );
    end

    % Throw out the first few data
    real_tdc_1 = real_tdc_1(num_frames_throw_away:end);
    real_tdc_2 = real_tdc_2(num_frames_throw_away:end);
    real_tac_1 = real_tac_1(num_frames_throw_away:end);
    real_tac_2 = real_tac_2(num_frames_throw_away:end);

    % Normalize the TAC data
    real_tac_1 = real_tac_1 - mean(real_tac_1(1:10));
    real_tac_2 = real_tac_2 - mean(real_tac_2(1:10));


    %-------- First fit the TDC into an exponential curve
    clear eerfh_1 fh_1 rel_1 t
    clear eerfh_2 fh_2 rel_2

    real_tdc_1=real_tdc_1( start_point(1):end_point(1) );
    real_tdc_2=real_tdc_2( start_point(2):end_point(2) );

    t = 1:size(real_tdc_1,2); 
    rel_1=real_tdc_1;
    rel_2=real_tdc_2;

    % Define the exponential function
      fh_1 = @(x,p) p(1) + p(2)*exp(-x./p(3));
      fh_2 = @(x,q) q(1) + q(2)*exp(-x./q(3));

    % Define the error function. this is the function to minimize
      errfh_1 = @(p,x,y) sum((y(:)-fh_1(x(:),p)).^2);
      errfh_2 = @(q,x,y) sum((y(:)-fh_1(x(:),q)).^2);

    % An initial guess of the exponential parameters
      p0 = [mean(rel_1) (max(rel_1)-min(rel_1)) (max(t) - min(t))/2];
      q0 = [mean(rel_2) (max(rel_2)-min(rel_2)) (max(t) - min(t))/2];

    % search for solution
      P = fminsearch(errfh_1,p0,[],t,rel_1); 
      Q = fminsearch(errfh_2,q0,[],t,rel_2); 

    % plot the result
     subplot(2,1,1);  plot(t,rel_1,'b',t,fh_1(t,P),'r-')
     title('finger 1')
     subplot(2,1,2); plot(t,rel_2,'b',t,fh_2(t,Q),'r-')
     title('finger 2')

     % Calculate the fitted value of tdc 
     clear real_tdc_fit_1 real_tdc_fit_2
     
     for j=1:size(real_tdc_1,2)
      real_tdc_fit_1(j)=P(1)+P(2)*exp(-j./P(3));
      real_tdc_fit_2(j)=Q(1)+Q(2)*exp(-j./Q(3));
     end


     %--------- Linear fits to get TAU and T_FINAL

    clear yfit_1 yfit_2

    real_tac_1=real_tac_1(start_point(1):end_point(1));
    real_tac_2=real_tac_2(start_point(2):end_point(2));

    R = polyfit(real_tac_1,real_tdc_fit_1,1);
    yfit_1 = R(1)*real_tac_1 + R(2);

    S = polyfit(real_tac_2,real_tdc_fit_2,1);
    yfit_2 = S(1)*real_tac_2 + S(2);

    subplot(2,1,1); plot(real_tac_1,real_tdc_fit_1,'b*',real_tac_1,yfit_1,'r-.');
    subplot(2,1,2); plot(real_tac_2,real_tdc_fit_2,'b*',real_tac_2,yfit_2,'r-.');

    % Invert values in the first colume of R/S to get positive TAU
    R(1) = -R(1);
    S(1) = -S(1);
    
    Trial = num2str(trial);
    
    fprintf(fileID,' %5f %3f   ', R, S);
    fprintf(fileID,'Trial # %s\n', Trial);
end


