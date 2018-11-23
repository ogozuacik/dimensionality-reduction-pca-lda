function [ accuracy, predict ] = predictOmer2( testData,testLabels,mu,Sigma,classCount)
    %prediction
    score=zeros(size(testData,1),classCount);
    predict=zeros(size(testData,1),1);
    for i=1:classCount
        score(:,i) = gaussOmer2(testData,mu(i,:),Sigma(:,:,i));
    end    
    for i=1:size(testData,1)
        predict(i)=find(score(i,:)==max(score(i,:)),1);
    end
    %correction of results
    predict=predict-1;
    result = (predict(:,1)==testLabels);
    accuracy = sum(double(result))/size(testData,1);
end

