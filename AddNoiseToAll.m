function [ result_matrix ] = AddNoiseToAll( input_matrix, d )
%AddNoisetoAll - add noise to all matrix elements
%   d - maximum change, range: (0,1)
%   input_matrix - matrix to which elements noise will be added

%%% generate noise
noise = rand(size(input_matrix)) .*2 -1;
noise = noise .*d;

%%% add noise
result_matrix = input_matrix + noise;

return