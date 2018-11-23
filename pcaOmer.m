function [ omega, kaiserLimit ] = pcaOmer( data )
%I used some parts of the code from my CS 464 homework related to pca 
    %finding eigenvalues and omega
    [omega, eigenValues] = eig(cov(data));
    eigenValues=diag(eigenValues);
    %sorting eigenvalues in descending order
    [eigenValues,index]=sort(eigenValues,'descend');
    kaiserLimit=find(eigenValues < 1,1) -1;
    omega=omega(:,index)';
    %plotting eigenvalues
    figure();plot(eigenValues)
    title('Eigenvalues in descending order')
    xlabel('Index'); ylabel('Eigenvalues')
end

