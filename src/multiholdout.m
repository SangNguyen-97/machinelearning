% Load datad
load Code/data.mat;

% Re-labeling
t_all.Object = removecats(t_all.Object);

% Partition data
cvpt = cvpartition(t_all.Object,"Holdout",0.2);
traindata = t_all(training(cvpt),:);
testdata = t_all(test(cvpt),:);
xtrain = traindata{:,1:end-1};
ytrain = traindata{:,end};
xtest = testdata{:,1:end-1};
ytest = testdata{:,end};

% ADDITIONAL: Transform data using FFT before feeding to PCA
xtrain = fftmat(xtrain);
xtest = fftmat(xtest);

% Feature reduction - PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(xtrain);

featnum = 18;
pca_xtrain = scoreTrain(:,1:featnum);

% Partition data and fit SVM model (Hold-out)
temp = templateSVM("KernelFunction","polynomial","Standardize",true);
mdl = fitcecoc(pca_xtrain,ytrain,"Learner",temp);

% Transform test data
pca_xtest = (xtest-mu)*coeff(:,1:featnum);

% Classify test data using trained model
pred = predict(mdl,pca_xtest);

% Calculating stats
iswrongneg = (pred ~= ytest) & (pred ~= "Object 1");
istrueneg = (pred == ytest) & (pred ~= "Object 1");
FN = sum(iswrongneg);
TN = sum(istrueneg);
iswrongpos = (pred ~= ytest) & (pred == "Object 1");
istruepos = (pred == ytest) & (pred == "Object 1");
FP = sum(iswrongpos);
TP = sum(istruepos);

FDR = FP / (FP + TP);
NPV = TN / (TN + FN);
TPR = TP / (TP + FN);
TNR = TN / (TN + FP);

F1 = (2*TP)/(2*TP + FP + FN);

% Misclassification Error
datasum = size(ytest,1);
hit = sum(pred == ytest);
missRate = 1 - hit/datasum;

% Summing up
sumup = table(TP,FP,TN,FN,FDR,NPV,TPR,TNR,F1,missRate)

%% PLOTTING AREA
% % PCA components' significance chart
% subplot(2,2,1);
% pareto(explained);
% title("Pareto chart of PCA");
% xlabel("Principle Components (PC)");
% 
% % Train data in reduced feature space
% subplot(2,2,2);
% gscatter(pca_xtrain(:,1),pca_xtrain(:,2),ytrain);
% title("Transformed feature space");
% xlabel("PC1");
% ylabel("PC2");

% Confusion chart
subplot(2,2,3);
confusionchart(ytest,pred);
title("Confusion matrix");

% % ROC curve
% mdl = fitPosterior(mdl);
% 
% [~,score_svm] = resubPredict(mdl);
% [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(ytrain,score_svm(:,1),"Object 1");
% subplot(2,2,4);
% plot(Xsvm,Ysvm);
% title("Receiver Operator Characteristic Curve");
% xlabel("False positive rate");
% ylabel("True positive rate");

%% AUXILIARY FUNCTION

% % Perform FFT on each row of the input matrix and return only 1/4 of the
% % FFT result for each row
% function retmat = fftmat(mat)
%     retmat_temp = fft(mat');
%     retmat = retmat_temp';
%     fin = ceil(size(mat,2)/4);
%     retmat = abs(retmat(:,1:fin));
% end
