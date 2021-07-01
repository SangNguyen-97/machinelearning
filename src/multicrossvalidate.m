% Loading pre-read data from Excel files
load Code/data.mat;

% Re-labeling
t_all.Object = removecats(t_all.Object);
order = unique(t_all.Object);

% Create CV partition
cvp = cvpartition(t_all.Object,"KFold",5);

% Defining handle function for CV
func = @(xtrain,ytrain,xtest,ytest) confusionmat(ytest,classf(xtrain,ytrain,xtest),"Order",order);

% % Compute CV error
% cvError = crossval('mcr',t_all{:,1:end-1},t_all.Object,'Predfun',@classf,'Partition',cvp);

% Compute confusion matrix
confMat = crossval(func,fftmat(t_all{:,1:end-1}),t_all.Object,"Partition",cvp);
cvMat = reshape(sum(confMat),5,5);
confusionchart(cvMat,order);

% Calculating stats
FN = sum(cvMat(1,2:end),"all");
TN = sum(cvMat(2:end,2:end),"all");
FP = sum(cvMat(2:end,1),"all");
TP = cvMat(1,1);

FDR = FP / (FP + TP);
NPV = TN / (TN + FN);
TPR = TP / (TP + FN);
TNR = TN / (TN + FP);
F1 = (2*TP)/(2*TP + FP + FN);

datasum = size(t_all,1);
hit = sum([cvMat(1,1) cvMat(2,2) cvMat(3,3) cvMat(4,4) cvMat(5,5) ],"all");
globalMissRate = 1 - hit/datasum;

%Summing up
sumup = table(TP,FP,TN,FN,FDR,NPV,TPR,TNR,F1,globalMissRate)

%% CLASSIFIER
function yfit = classf(xtrain,ytrain,xtest)
featnum = 18;

% Normalized training data and do PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(xtrain);

% Pick transformed features
pca_xtrain = scoreTrain(:,1:featnum);

% Fit classification model
temp = templateSVM("KernelFunction","polynomial","Standardize",true);
mdl = fitcecoc(pca_xtrain,ytrain,"Learner",temp);

% Classify test data using trained model
pca_xtest = (xtest - mu)*coeff(:,1:featnum);
yfit = predict(mdl,pca_xtest);
end

%% AUXILIARY FUNCTION

% % Perform FFT on each row of the input matrix and return only 1/4 of the
% % FFT result for each row
% function retmat = fftmat(mat)
%     retmat_temp = fft(mat');
%     retmat = retmat_temp';
%     fin = ceil(size(mat,2)/4);
%     retmat = abs(retmat(:,1:fin));
% end
