%% Load data from .h5 files
clear all_data
all_data = process_h5_file('acrylic_211.h5');


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




for trial=1:10
    total_samples = length(all_data(trial).data.biotac(1).pdc);

    % find the squeeze moment
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
 
 
    % find the holding moment
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


%% Exponential fits of TDC for SQEEZE, HOLD, SLOW SLIDE, and FAST SLIDE


area_squeeze=zeros(10,2);
area_hold=zeros(10,2);
area_slow_slide=zeros(10,2);
area_fast_slide=zeros(10,2);

hold_pressure=zeros(10,2);
slow_slide_pressure=zeros(10,2);
fast_slide_pressure=zeros(10,2);

SQUEEZE_1 = zeros(10,3);
SQUEEZE_2 = zeros(10,3);


%% SQUEEZE
for trial=1:10
   
    clear tdc_1 tdc_2 
    clear eerfh_1 fh_1 rel_1 t
    clear eerfh_2 fh_2 rel_2

    tdc_1=all_data(trial).data.biotac(1).tdc( (sqz_start(trial)+150):(sqz_stop(trial)-150) );
    tdc_2=all_data(trial).data.biotac(2).tdc( (sqz_start(trial)+150):(sqz_stop(trial)-150) );
    
    t = 1:size(tdc_1,1); 
    rel_1 = tdc_1;
    rel_2 = tdc_2;

    % Define the exponential function
      fh_1 = @(x,p) p(1) + p(2)*exp(-x./p(3));
      fh_2 = @(x,q) q(1) + q(2)*exp(-x./q(3));

    % Define the error function. this is the function to minimize
      errfh_1 = @(p,x,y) sum((y(:)-fh_1(x(:),p)).^2);
      errfh_2 = @(q,x,y) sum((y(:)-fh_1(x(:),q)).^2);

    % An initial guess of the exponential parameters
      p0 = [mean(rel_1) ( max(rel_1)-min(rel_1) ) (max(t) - min(t))/2];
      q0 = [mean(rel_2) ( max(rel_2)-min(rel_2) ) (max(t) - min(t))/2];

    % search for solution
      SQUEEZE_1(trial,:) = fminsearch(errfh_1,p0,[],t,rel_1); 
      SQUEEZE_2(trial,:) = fminsearch(errfh_2,q0,[],t,rel_2); 

    % plot the result
     subplot(2,1,1);  plot(t,rel_1,'b',t,fh_1(t,SQUEEZE_1(trial,:)),'r-'); hold on
     title('finger 1')
     subplot(2,1,2); plot(t,rel_2,'b',t,fh_2(t,SQUEEZE_2(trial,:)),'r-');   hold on
     title('finger 2')

end


%% HOLD
HOLD_1 = zeros(10,3);
HOLD_2 = zeros(10,3);

for trial=1:10
%  %%
%  trial = 1;
    clear tdc_1 tdc_2 
    clear eerfh_1 fh_1 rel_1 t
    clear eerfh_2 fh_2 rel_2

    tdc_1=all_data(trial).data.biotac(1).tdc( (hold_start(trial)+200):hold_stop(trial) );
    tdc_2=all_data(trial).data.biotac(2).tdc( (hold_start(trial)+200):hold_stop(trial) );
    
    t = 1:size(tdc_1,1); 
    rel_1 = tdc_1;
    rel_2 = tdc_2;

    % Define the exponential function
      fh_1 = @(x,p) p(1) + p(2)*exp(-x./p(3));
      fh_2 = @(x,q) q(1) + q(2)*exp(-x./q(3));

    % Define the error function. this is the function to minimize
      errfh_1 = @(p,x,y) sum((y(:)-fh_1(x(:),p)).^2);
      errfh_2 = @(q,x,y) sum((y(:)-fh_1(x(:),q)).^2);

    % An initial guess of the exponential parameters
      p0 = [mean(rel_1) ( max(rel_1)-min(rel_1) ) (max(t) - min(t))/2];
      q0 = [mean(rel_2) ( max(rel_2)-min(rel_2) ) (max(t) - min(t))/2];

    % search for solution
      HOLD_1(trial,:) = fminsearch(errfh_1,p0,[],t,rel_1); 
      HOLD_2(trial,:) = fminsearch(errfh_2,q0,[],t,rel_2); 

    % plot the result
     subplot(2,1,1);  plot(t,rel_1,'b',t,fh_1(t,HOLD_1(trial,:)),'r-'); hold on
     title('finger 1')
     subplot(2,1,2); plot(t,rel_2,'b',t,fh_2(t,HOLD_2(trial,:)),'r-');   hold on
     title('finger 2')

end


%% SLOW SLIDE
SLOW_SLIDE_1 = zeros(10,3);
SLOW_SLIDE_2 = zeros(10,3);

for trial=1:10

    clear tdc_1 tdc_2 
    clear eerfh_1 fh_1 rel_1 t
    clear eerfh_2 fh_2 rel_2

    tdc_1=all_data(trial).data.biotac(1).tdc( (slow_slide_start(trial)+300):(slow_slide_stop(trial)-100) );
    tdc_2=all_data(trial).data.biotac(2).tdc( (slow_slide_start(trial)+300):(slow_slide_stop(trial)-100) );
    
    t = 1:size(tdc_1,1); 
    rel_1 = tdc_1;
    rel_2 = tdc_2;

    % Define the exponential function
      fh_1 = @(x,p) p(1) + p(2)*exp(-x./p(3));
      fh_2 = @(x,q) q(1) + q(2)*exp(-x./q(3));

    % Define the error function. this is the function to minimize
      errfh_1 = @(p,x,y) sum((y(:)-fh_1(x(:),p)).^2);
      errfh_2 = @(q,x,y) sum((y(:)-fh_1(x(:),q)).^2);

    % An initial guess of the exponential parameters
      p0 = [mean(rel_1) ( max(rel_1)-min(rel_1) ) (max(t) - min(t))/2];
      q0 = [mean(rel_2) ( max(rel_2)-min(rel_2) ) (max(t) - min(t))/2];

    % search for solution
      SLOW_SLIDE_1(trial,:) = fminsearch(errfh_1,p0,[],t,rel_1); 
      SLOW_SLIDE_2(trial,:) = fminsearch(errfh_2,q0,[],t,rel_2); 

    % plot the result
     subplot(2,1,1);  plot(t,rel_1,'b',t,fh_1(t,SLOW_SLIDE_1(trial,:)),'r-'); hold on
     title('finger 1')
     subplot(2,1,2); plot(t,rel_2,'b',t,fh_2(t,SLOW_SLIDE_2(trial,:)),'r-');   hold on
     title('finger 2')

end



%% FAST SLIDE
FAST_SLIDE_1 = zeros(10,3);
FAST_SLIDE_2 = zeros(10,3);

for trial=1:10

    clear tdc_1 tdc_2 
    clear eerfh_1 fh_1 rel_1 t
    clear eerfh_2 fh_2 rel_2

    tdc_1=all_data(trial).data.biotac(1).tdc( (slow_slide_start(trial)):(slow_slide_stop(trial)) );
    tdc_2=all_data(trial).data.biotac(2).tdc( (slow_slide_start(trial)):(slow_slide_stop(trial)) );
    
    t = 1:size(tdc_1,1); 
    rel_1 = tdc_1;
    rel_2 = tdc_2;
    
    % Define the exponential function
      fh_1 = @(x,p) p(1) + p(2)*exp(-x./p(3));
      fh_2 = @(x,q) q(1) + q(2)*exp(-x./q(3));

    % Define the error function. this is the function to minimize
      errfh_1 = @(p,x,y) sum((y(:)-fh_1(x(:),p)).^2);
      errfh_2 = @(q,x,y) sum((y(:)-fh_1(x(:),q)).^2);

    % An initial guess of the exponential parameters
      p0 = [mean(rel_1) ( max(rel_1)-min(rel_1) ) (max(t) - min(t))/2];
      q0 = [mean(rel_2) ( max(rel_2)-min(rel_2) ) (max(t) - min(t))/2];

    % search for solution
      FAST_SLIDE_1(trial,:) = fminsearch(errfh_1,p0,[],t,rel_1); 
      FAST_SLIDE_2(trial,:) = fminsearch(errfh_2,q0,[],t,rel_2); 

    % plot the result
     subplot(2,1,1);  plot(t,rel_1,'b',t,fh_1(t,FAST_SLIDE_1(trial,:)),'r-'); hold on
     title('finger 1')
     subplot(2,1,2); plot(t,rel_2,'b',t,fh_2(t,FAST_SLIDE_2(trial,:)),'r-');   hold on
     title('finger 2')

end
