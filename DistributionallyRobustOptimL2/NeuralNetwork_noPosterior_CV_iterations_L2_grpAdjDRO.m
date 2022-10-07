%squaring peffects - this one is not optimal 
%continous distribution for probable peak selection 
function [] = NeuralNetwork_noPosterior_CV_iterations_L2_grpAdjDRO()


%read in non-posterior dataset: 
%n=102 spectra, 3409 m/z locations 
% test_prod_50=readtable('PreprocessNoPosteriorData/SmallGeneticMassSpecSpectraOnly_TEST_shuff50.csv');  %testClust_1_meanEffAllSamp
% %convert table to array/matrix as otherwise cannot do mat'
% test_prod_50 = table2array(test_prod_50);

% mz = test_prod_50(1,:);
% test_prod_50 = test_prod_50(2:end,:);

% save('PreprocessNoPosteriorData/test_prod_50.mat', 'test_prod_50');

load('PreprocessNoPosteriorData/test_prod_50.mat');
%rerun this after opening the above .mat - as it contains label at last
%column
test_labels_50 = test_prod_50(:,end);
test_prod_50 = test_prod_50(:,1:end-1);

% train_prod_50=readtable('PreprocessNoPosteriorData/SmallGeneticMassSpecSpectraOnly_TRAIN_shuff50.csv');
% 
% train_prod_50 = table2array(train_prod_50);
% 
% mz = train_prod_50(1,:);
% train_prod_50 = train_prod_50(2:end,:);
% 
% save('PreprocessNoPosteriorData/train_prod_50.mat', 'train_prod_50');

load('PreprocessNoPosteriorData/train_prod_50.mat');

%rerun this after opening the above .mat - as it contains label at last
%column
train_labels_50 = train_prod_50(:,end);
train_prod_50 = train_prod_50(:,1:end-1);

% Path to code so we can run all runs in the expendable "model_outputs" directory.
path(path, '../PoissonRLD_spectrometry/')
cd model_outputs



%% Initialize parameters
%desired_error= 1e-1;  % 1e-3;
%desired_error= 0.325;
desired_error= 0.16;
%Learning_Rate= 0.1;  %0.1;
%Learning_Rate= 0.325;
Learning_Rate= 0.16;
%hidden_layers=[102 25 1];
%hidden_layers=[3409 102 1];
hidden_layers=[12];  %12 - AUC 0.6026
plotting='yes';
lambda = 2^-17;

%add MLP code to paths 
path(path, 'MLP_modified/');

%random seed 
rng(7);

%% Training
[net]=BP_TB_L2_groupAdj(train_prod_50,train_labels_50,desired_error,Learning_Rate,hidden_layers,plotting,lambda);


%%%%%%%%%%% prediction
%% Prediction 
[predictedTest_50]=predict(net,test_prod_50);


combActualPred = horzcat(test_labels_50, predictedTest_50);


[X2,Y2,T,AUC] = perfcurve(test_labels_50, predictedTest_50, 1);


%extract groups separately to compare AUC
%positive class
posGroup = combActualPred(combActualPred(:,1) == 1, :);

%calculate group mse 
%calculate mse
MSE = mean(posGroup(:,1) - posGroup(:,2)).^2;
display(MSE);

%calculate accuracy 
%convert probability to class at prob threshold > 0.5
posGroup(posGroup(:,2) > 0.5, 2) = 1;
posGroup(posGroup(:,2) < 0.5, 2) = 0;

acc_count = nnz(posGroup(:,1) == posGroup(:,2)); 
acc = acc_count/length(test_labels_50);
display(acc);

%Repeat for neg group %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%positive class for pos group
negGroup = combActualPred(combActualPred(:,1) == 0, :);

%calculate group mse 
%calculate mse
MSE = mean(negGroup(:,1) - negGroup(:,2)).^2;
display(MSE);

%calculate accuracy
%convert probability to class at prob threshold > 0.5
negGroup(negGroup(:,2) > 0.5, 2) = 1;
negGroup(negGroup(:,2) < 0.5, 2) = 0;

acc_count = nnz(negGroup(:,1) == negGroup(:,2)); 
acc = acc_count/length(test_labels_50);
display(acc);



figure()
plot(X2,Y2)
title(['ROC with AUC: ',num2str(AUC)])
xlabel("False Positive Rate: (1 - Specificity)")
ylabel("True Positive Rate: Sensitivity")

%mean squared error - note accuracy cannot be calculated for probability
MSE = mean((test_labels_50 - predictedTest_50).^2);
display(MSE);


end

