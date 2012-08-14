function [ centered_signal ] = select_signal( signal )
%select_signal Given a signal, will take the signal and segment out the
%area where we care about only.  It basically removes the area of the
%signal that the object is not doing anything

all_signals = mean(abs(signal),2);
%all_signals = max(abs(signal),2);

idx = find(all_signals > 10);

if (isempty(idx))
    idx = find(all_signals > 3);
end

start_sig = max(1,idx(1)-10);
end_sig = min(idx(end)+10,size(all_signals,1));

centered_signal = signal(start_sig:end_sig,:);

end

