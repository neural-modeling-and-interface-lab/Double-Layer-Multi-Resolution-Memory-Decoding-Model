%% This script is used for checking the performance of all compared algorithms
% Corresponding to Appendix. B in the manuscript of NECO-2021-She
% Project: RAM USC
% Author: Xiwei She
% Date: 2021-05-24
clear;clc;

Partition = 1:10; % 1:10

%% Start processing
saveCase = 'Synthetic';
patientIndex = 1:3;
categoryIndex = 1;

% These are the results of the proposed model, can be loaded/read from
% Synthetic_4_MeasuringPerformance.m with corresponding datasets
MCC_DLMR = [0.8143; 0.5345; 1]; STD_DLMR = [0.0359; 0.0624; 0];

% DLMR without bagging
MCC_noBagging = zeros(length(patientIndex), length(categoryIndex)); STD_noBagging = zeros(length(patientIndex), length(categoryIndex));
% Naive Bayes Decoder
MCC_NB = zeros(length(patientIndex), length(categoryIndex)); STD_NB = zeros(length(patientIndex), length(categoryIndex));
% Logistice Regression
MCC_LR = zeros(length(patientIndex), length(categoryIndex)); STD_LR = zeros(length(patientIndex), length(categoryIndex));
% LASSO with original spikes
MCC_LASSOSPK = zeros(length(patientIndex), length(categoryIndex)); STD_LASSOSPK = zeros(length(patientIndex), length(categoryIndex));
% LASSO with PCA of original spikes
MCC_LASSOPCA = zeros(length(patientIndex), length(categoryIndex)); STD_LASSOPCA = zeros(length(patientIndex), length(categoryIndex));

warning off;
paIndex = 0;
for pa = patientIndex
    paIndex = paIndex + 1;
    DataFolder = comparisonSubjects(pa);
    
    caIndex = 0;
    for ca = categoryIndex
        caIndex = caIndex + 1;
        caTemp = ca;
        decodingCategory = comparisonCategories(caTemp);
        
        disp(['Processing Data of ', DataFolder, ' !']);
        % Get the overall concantenent target vector
        overallTarget = [];
        for currentPartition = Partition
            % Load models results
            load(['Results\AlgorithmComparison\', DataFolder, '\AlgorithmComparison_', DataFolder,'_part', mat2str(currentPartition), '_rep.mat']);
            overallTarget = [overallTarget; MDfit.TestingSet_target];
        end
        repeatSize = size(MDfit.R_exp, 1);
        resolutionPool = MDfit.m_all;
        clear MDfit
        
        % Algorithms for comparison
        overallProb_0 = zeros(repeatSize, length(overallTarget)); % Lasso without bagging
        overallProb_1 = zeros(repeatSize, length(overallTarget)); % Logistic regression
        overallProb_2 = zeros(repeatSize, length(overallTarget)); % Naive Bayes
        overallProb_3 = zeros(repeatSize, length(overallTarget)); % Lasso with original spikes
        overallProb_4 = zeros(repeatSize, length(overallTarget)); % Lasso with spike pca
        
        % Different repeat times
        for repeat = 1:repeatSize
            % Different partitions
            partitionIndexStart = 1;
            for currentPartition = Partition
                
                % Load models results
                load(['Results\AlgorithmComparison\', DataFolder, '\AlgorithmComparison_', DataFolder,'_part', mat2str(currentPartition), '_rep.mat']);
                
                %% Results processing
                % Index for saving
                partitionIndexEnd = partitionIndexStart + size(MDfit.R_exp(repeat, 1).prob_0, 1) - 1;
                
                % Predictions - algorithms
                selectedResolution = randi(length(MDfit.R_exp));
                overallProb_0(repeat, partitionIndexStart:partitionIndexEnd) = MDfit.R_exp(repeat, end).prob_0;
                overallProb_1(repeat, partitionIndexStart:partitionIndexEnd) = MDfit.R_exp(repeat, end).prob_1;
                overallProb_2(repeat, partitionIndexStart:partitionIndexEnd) = MDfit.R_exp(repeat, end).prob_2(:, 1);
                overallProb_3(repeat, partitionIndexStart:partitionIndexEnd) = MDfit.R_exp(repeat, end).prob_3;
                overallProb_4(repeat, partitionIndexStart:partitionIndexEnd) = MDfit.R_exp(repeat, end).prob_4;
                
                % Index for saving
                partitionIndexStart = partitionIndexEnd + 1;
            end
        end
        
        disp('========================== The Overall Performance =========================');
        if sum(overallTarget(:)) <= 2
            disp(['This ', DataFolder,' & ', decodingCategory,' doesnt contain enough categorical trials']);
        else
            
            %% Get and Show Performance
            overallPred_0 = double( overallProb_0 > 0.5 );
            overallPred_1 = double( overallProb_1 > 0.5 );
            overallPred_2 = double( overallProb_2 > 0.5 );
            overallPred_3 = double( overallProb_3 > 0.5 );
            overallPred_4 = double( overallProb_4 > 0.5 );
            
            MCC_0 = zeros(size(overallPred_0, 1), 1);
            MCC_1 = zeros(size(overallPred_1, 1), 1);
            MCC_2 = zeros(size(overallPred_2, 1), 1);
            MCC_3 = zeros(size(overallPred_3, 1), 1);
            MCC_4 = zeros(size(overallPred_4, 1), 1);
            for fold = 1:size(overallPred_0, 1)
                CM_0 = confusionmat(overallPred_0(fold, :), overallTarget); MCC_0(fold, 1) = mcc(CM_0);
                CM_1 = confusionmat(overallPred_1(fold, :), overallTarget); MCC_1(fold, 1) = mcc(CM_1);
                CM_2 = confusionmat(overallPred_2(fold, :), overallTarget); MCC_2(fold, 1) = mcc(CM_2);
                CM_3 = confusionmat(overallPred_3(fold, :), overallTarget); MCC_3(fold, 1) = mcc(CM_3);
                CM_4 = confusionmat(overallPred_4(fold, :), overallTarget); MCC_4(fold, 1) = mcc(CM_4);
            end
            
            mMCC_0 = mat2str(mean(MCC_0)); STD_0 = mat2str(std(MCC_0));
            mMCC_1 = mat2str(mean(MCC_1)); STD_1 = mat2str(std(MCC_1));
            mMCC_2 = mat2str(mean(MCC_2)); STD_2 = mat2str(std(MCC_2));
            mMCC_3 = mat2str(mean(MCC_3)); STD_3 = mat2str(std(MCC_3));
            mMCC_4 = mat2str(mean(MCC_4)); STD_4 = mat2str(std(MCC_4));
            
            disp(['MCC_0=', mMCC_0, '(', STD_0, ')',  ' MCC_1=', mMCC_1, '(', STD_1, ')', ' MCC_2=', mMCC_2, '(', STD_2, ')', ' MCC_3=', mMCC_3, '(', STD_3, ')', ' MCC_4=', mMCC_4, '(', STD_4, ')']);
        end
        MCC_noBagging(paIndex, caIndex) = str2double(mMCC_0); STD_noBagging(paIndex, caIndex) = str2double(STD_0);
        MCC_NB(paIndex, caIndex) = str2double(mMCC_2); STD_NB(paIndex, caIndex) = str2double(STD_2);
        MCC_LR(paIndex, caIndex) = str2double(mMCC_1); STD_LR(paIndex, caIndex) = str2double(STD_1);
        MCC_LASSOSPK(paIndex, caIndex) = str2double(mMCC_3); STD_LASSOSPK(paIndex, caIndex) = str2double(STD_3);
        MCC_LASSOPCA(paIndex, caIndex) = str2double(mMCC_4); STD_LASSOPCA(paIndex, caIndex) = str2double(STD_4);
    end
end

%% Visualization - Algorithm Comparisons

MCCs = [MCC_DLMR, MCC_NB, MCC_LR, MCC_LASSOSPK, MCC_LASSOPCA];
STDs = [STD_DLMR, STD_NB, STD_LR, STD_LASSOSPK, STD_LASSOPCA];
figure;
barwitherr(STDs, MCCs)
legend('DLMR', 'Naive Bayes', 'Logistic Regression', 'Lasso with spikes', 'Lasso with PCA of spikes')