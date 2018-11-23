function gaussPdf = gaussOmer2(X, mu, sigma)
    Xmu = X-mu;
    gaussPdf = 1/(sqrt(det(sigma))*(2*pi)^(size(X,2)/2)) * exp(-0.5*diag(Xmu*pinv(sigma)*Xmu'));
end