function [ mu, sigma ] = gaussClassFitOmer( data )
    mu = mean(data,1);
    Xm = data - mu;
    sigma = (Xm' * Xm)/size(data,1);
end