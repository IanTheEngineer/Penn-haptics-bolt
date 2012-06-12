function [ handles ] = getJButtonHandles( names )
%getJButtonHandles Given an array of names, return jButton handles.  Used
%in conjunction with findobj.  Goes through the GUI and returns the handles
%to the GUI objects.

% Author: Vivian Chu
% University of Pennsylvania
% email: chuv@grasp.upenn.edu
% Website: http://www.bolt-haptics.seas.upenn.edu
% June 2012; Last revision: 12-June-2012

% Assume in the form {'1', '2', '3', etc}

    for i = 1:numel(names)
        handles{i} = findjobj('property',{'Text',names{i}},'property',{'UIClassID','ButtonUI'},'persist'); 
    end

end

