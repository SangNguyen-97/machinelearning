% Loading pre-read data from Excel files
load Code/data.mat;

% Re-labeling
t_all.Object(t_all.Object~="Object 1") = "Not Object 1";
t_all.Object = removecats(t_all.Object);
order = unique(t_all.Object);

% Create CV partition
cvp = cvpartition(t_all.Object,"KFold",5); % PARAMETER: Partition ratio for cross validation

% Defining handle function for CV
func = @(xtrain,ytrain,xtest,ytest) confusionmat(ytest,classf(xtrain,ytrain,xtest),"Order",order);

% % Compute CV error
% cvError = crossval('mcr',t_all{:,1:end-1},t_all.Object,'Predfun',@classf,'Partition',cvp);

% Compute confusion matrix
confMat = crossval(func,t_all{:,1:end-1},t_all.Object,"Partition",cvp);
cvMat = reshape(sum(confMat),2,2);
confusionchart(cvMat,order);

% Calculating stats
FN = cvMat(1,2);
TN = cvMat(2,2);
FP = cvMat(2,1);
TP = cvMat(1,1);

FDR = FP / (FP + TP);
NPV = TN / (TN + FN);
TPR = TP / (TP + FN);
TNR = TN / (TN + FP);
F1 = (2*TP)/(2*TP + FP + FN);
missRate = (FP + FN)/(TP + TN + FP + FN);

% Summing up
sumup = table(TP,FP,TN,FN,FDR,NPV,TPR,TNR,F1,missRate)


function yfit = classf(xtrain,ytrain,xtest)
featnum = 8; % PARAMETER: number of PCA feature to use for classification

% Normalized training data and do PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(xtrain);

% Pick transformed features
pca_xtrain = scoreTrain(:,1:featnum);

% Fit classification model
kernel = "polynomial"; % PARAMETER: type of kernel for SVM
mdl = fitcsvm(pca_xtrain,ytrain,"KernelFunction",kernel,"Standardize",true);

% Classify test data using trained model
pca_xtest = (xtest - mu)*coeff(:,1:featnum);
yfit = predict(mdl,pca_xtest);
end

%% AUXILIARY FUNCTION

% Perform FFT on each row of the input matrix and return only 1/4 of the
% FFT result for each row
function retmat = fftmat(mat)
    retmat_temp = fft(mat');
    retmat = retmat_temp';
    fin = ceil(size(mat,2)/4);
    retmat = abs(retmat(:,1:fin));
end
