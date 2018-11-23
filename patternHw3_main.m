clear; clc;
%importing data
workspace = importdata('digits.mat');
data = [workspace.digits workspace.labels];
clear workspace
%randomizing the order in data
data = data(randperm(size(data,1)),:);
%data(:,1:400)= (data(:,1:400) - (min(data(:,1:400))))./ (max(data(:,1:400)) - min(data(:,1:400)));

%seperating training and test classses
trainData = data(1:2500,1:401);
testData = data(2501:5000,1:401);
%clear data

%% Question 1
mu = zeros(10,400);
sigma = zeros(400,400,10);
for i=0
    %taking only i'th digit
    temp = trainData(trainData(:,401)==i,:);
    temp = temp(:,1:400);
    [mu(i+1,:),sigma(:,:,i+1)] = gaussClassFitOmer(temp); 
end

%% Question 2

[dataPCA, kaiserPCA_Limit] = pcaOmer(data(:,1:400)); 
sampleMean = mean(data(:,1:400));

%with respect to the plot&according to the Kaiser rule
%principal top principal components are plotted
for i=1:kaiserPCA_Limit
    I=dataPCA(i,:);
    figure, imagesc(reshape(I,20,20));
    title(strcat('Principal Component=',num2str(i)))
    colormap(gray);
    axis image;
end

%plotting sample mean
I=sampleMean;
figure, imagesc(reshape(I,20,20));
title("Sample Mean")
colormap(gray);
axis image;


%% Question 2---cont
%fitting gaussians for different # of principal components
x = 1:10:200;
testAccuracy=[]; trainAccuracy=[];
for j=x
    mu = zeros(10,j);
    sigma = zeros(j,j,10);
    subSpaceTest = testData(:,1:400) * dataPCA(1:j,:)';
    subSpaceTrain = trainData(:,1:400) * dataPCA(1:j,:)';
    for i=1:10
        %taking only i-1'th digit
        temp = subSpaceTrain(trainData(:,401)== i-1,:);
        [mu(i,:),sigma(:,:,i)] = gaussClassFitOmer(temp); 
    end
    testAccuracy = [testAccuracy predictOmer2(subSpaceTest,testData(:,401),mu,sigma,10)];
    trainAccuracy = [trainAccuracy predictOmer2(subSpaceTrain,trainData(:,401),mu,sigma,10)];
end
figure
plot(x,trainAccuracy,'DisplayName','trainAccuracy'); hold on;
plot(x,testAccuracy,'DisplayName','testAccuracy'); legend show;
title('Subspace dimension vs Train&Test Accuracy with PCA'); 
xlabel('Subspace dimension'); ylabel('Accuracy')

clear subSpaceTrain; clear subSpaceTest; clear mu; clear sigma;clear temp

%% Question 3--LDA

testAccuracy=zeros(9,1);
trainAccuracy=zeros(9,1);
for j=1:9
    [ldaMAP,ldaBASES]=lda(data(:,1:400),data(:,401),j);
    trainLDA=ldaMAP(1:2500,:);
    testLDA=ldaMAP(2501:5000,:);
    mu = zeros(10,j);
    sigma = zeros(j,j,10);
    for i=1:10
        temp = trainLDA(trainData(:,401)== i-1,:);
        [mu(i,:),sigma(:,:,i)] = gaussClassFitOmer(temp);
    end
    testAccuracy(j)=predictOmer2(testLDA,testData(:,401),mu,sigma,10);
    trainAccuracy(j)=predictOmer2(trainLDA,trainData(:,401),mu,sigma,10);
end

%images of the new bases
for i=1:9
    I=ldaBASES.M(:,i);
    figure, imagesc(reshape(I,20,20));
    title(strcat('Base=',num2str(i)))
    colormap(gray);
    axis image;
end

%plotting accuracies
figure
plot(1:1:9,trainAccuracy,'DisplayName','trainAccuracy'); hold on;
plot(1:1:9,testAccuracy,'DisplayName','testAccuracy'); legend show;
title('Subspace dimension vs Train&Test Accuracy with LDA'); 
xlabel('Subspace dimension'); ylabel('Accuracy')

clear ldaBASES; clear ldaMAP; clear testLDA; clear trainLDA


