function [ coefficients ] = find_poly_coeff( signal )
%find_poly_coeff Given a electrode signal will return coefficients for a
%fitted line to all signals

degree = 3;

coefficients = ones(size(signal,2),degree+1);

for i = 1:size(signal,2)
    coefficients(i,:) = polyfit(1:size(signal,1),signal(:,i)',degree);
end

end