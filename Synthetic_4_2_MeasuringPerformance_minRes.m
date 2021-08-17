%% This script is used for adding the MDM results of very small b-spline resolution
% original modeled resolution [5:25, 50:5:150] (Syntehtic_4_MeasuringPerformance.m)
% new added resolution [0:4]
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-11-26
clear;clc;



%% Uncomment/comment corresponding cases you want to check

% Case 1: single neuron + low resolution
runCase = 'SN&LS4';
Subject = 'SingleNeuron';

% % Case 2: single neuron + high resolution
% runCase = 'SN&HS4';
% Subject = 'SingleNeuron';

% % Case 3: two neurons
% runCase = 'CN&CS4';
% Subject = 'CombinedNeuron';

nCategories = 2;
Partition = 1:10; % 10-fold nested cross-validation

%% Start processing
warning off;
for ca = 1
    switch (ca)
        case 1
            category = 'A';
        case 2
            category = 'B';
        case 3
            category = 'C';
    end
    
    disp(['Processing Data of ', category,'!']);
    % Get the overall concantenent target vector
    overallTarget = [];
    for currentPartition = Partition
        % Load models of two layers
        load(['Results\',Subject, '&', mat2str(nCategories),'Categories\Raw Results\NestedCVDLMDM_synthetic_',Subject, '&', mat2str(nCategories),'Categories_', category,'_part', mat2str(currentPartition),'_',runCase,'_minRes.mat']);
        overallTarget = [overallTarget; MDfit.TestingSet_target];
    end
    repeatSize = size(MDfit.R_first, 1);
    numFold = length(MDfit.R_first(1, 1).FL_outside_MCCs);
    numKnots = size(MDfit.R_first, 2);
    resolutionPool = MDfit.m_all;
    BSplineOrder = MDfit.d;
    clear MDfit
    
    % First Layer outputs
    overallPredictions_firstLayer = zeros(repeatSize, numFold, length(overallTarget));
    overallPredictions_firstLayer_resolutions = zeros(repeatSize, numFold, length(overallTarget), numKnots);
%     FL_MCCs = zeros(repeatSize, numFold, numKnots, numFold);
    
    % Second Layer outputs
    overallPredictions_secondLayer = zeros(repeatSize, numFold, length(overallTarget));
    overallProbabilities_secondLayer = zeros(repeatSize, numFold, length(overallTarget));
    
    % Different repeat times
    for repeat = 1:repeatSize
        % Different partitions
        partitionIndexStart = 1;
        for currentPartition = Partition
            
            % Load models of two layers
            load(['Results\',Subject, '&', mat2str(nCategories),'Categories\Raw Results\NestedCVDLMDM_synthetic_',Subject, '&', mat2str(nCategories),'Categories_', category,'_part', mat2str(currentPartition), '_',runCase,'_minRes.mat']);
            
            %% First&Second Layer processing
            % Index for saving
            partitionIndexEnd = partitionIndexStart + size(MDfit.R_first(repeat, 1).FL_outside_predictions, 2) - 1;
            % Find the best inside resolution
            bestInsideMCC = 0;
            bestInsideResolution = 1;
            for resolution = 1:length(resolutionPool)
                currentInsideMCC = MDfit.R_first(repeat, resolution).FL_inside_MCC;
%                 FL_MCCs(repeat, currentPartition, resolution, :) = MDfit.R_first(repeat, resolution).FL_outside_MCCs;
                overallPredictions_firstLayer_resolutions(repeat, 1:size(MDfit.R_first(repeat, resolution).FL_outside_predictions, 1), partitionIndexStart:partitionIndexEnd, resolution) = MDfit.R_first(repeat, resolution).FL_outside_predictions;
                if currentInsideMCC > bestInsideMCC
                    bestInsideMCC = currentInsideMCC;
                    bestInsideResolution = resolution;
                end
            end
            % Predictions - first layer
            overallPredictions_firstLayer(repeat, 1:size(MDfit.R_first(repeat, bestInsideResolution).FL_outside_predictions, 1), partitionIndexStart:partitionIndexEnd) = MDfit.R_first(repeat, bestInsideResolution).FL_outside_predictions;
            
            % Predictions - second layer
            overallPredictions_secondLayer(repeat, 1:size(MDfit.R_first(repeat, bestInsideResolution).FL_outside_predictions, 1), partitionIndexStart:partitionIndexEnd) = MDfit.R_second(repeat).SL_outside_predictionss;
            % Probabilities - second layer
            overallProbabilities_secondLayer(repeat, 1:size(MDfit.R_first(repeat, bestInsideResolution).FL_outside_predictions, 1), partitionIndexStart:partitionIndexEnd) = MDfit.R_second(repeat).SL_outside_probabilities;
            
            % Index for saving
            partitionIndexStart = partitionIndexEnd + 1;
        end
    end
    
    disp('========================== The Overall Performance =========================');
    if sum(overallTarget(:)) <= 2
        disp(['This ', DataFolder,' doesnt contain enough categorical trials']);
    else
        %% First Layer Performance
        FL_overallMCC = zeros(repeatSize, numFold);
        FL_overallMCC_resolutions = zeros(repeatSize, numFold, numKnots);
        for repeat = 1:repeatSize
            
            tempPredictions_firstLayer = squeeze(overallPredictions_firstLayer(repeat, :, :));
            tempPredictions_firstLayer_resolutions = squeeze(overallPredictions_firstLayer_resolutions(repeat, :, :, :));
            % For each model from folds
            for fold = 1:size(tempPredictions_firstLayer, 1)
                tempModelPrediction_firstLayer = tempPredictions_firstLayer(fold, :);
                tempModelPrediction_firstLayer_resolutions = squeeze(tempPredictions_firstLayer_resolutions(fold, :, :));
                
                CM_fold_firstLayer = confusionmat(tempModelPrediction_firstLayer, overallTarget);
                if (size(CM_fold_firstLayer,1)==1&&size(CM_fold_firstLayer,2)==1)
                    CM_fold_firstLayer = [CM_fold_firstLayer(1,1) 0;0 0];
                end
                MCC_fold_firstLayer = mcc(CM_fold_firstLayer);
                FL_overallMCC(repeat, fold) = MCC_fold_firstLayer;
                
                % For each resolution
                for resolution = 1:length(resolutionPool)
                    CM_fold_firstLayer_resolution = confusionmat(tempModelPrediction_firstLayer_resolutions(:, resolution), overallTarget);
                    if (size(CM_fold_firstLayer_resolution,1)==1&&size(CM_fold_firstLayer_resolution,2)==1)
                        CM_fold_firstLayer_resolution = [CM_fold_firstLayer_resolution(1,1) 0;0 0];
                    end
                    MCC_fold_firstLayer_resolution = mcc(CM_fold_firstLayer_resolution);
                    FL_overallMCC_resolutions(repeat, fold, resolution) = MCC_fold_firstLayer_resolution;
                end
            end
        end
        
        % Visualize the MCC curve along with resolutions
        meanMCC_firstLayer_resolutions = squeeze(mean(squeeze(mean(FL_overallMCC_resolutions, 1)), 1));
        stdMCC_firstLayer_resolutions = squeeze(mean(squeeze(std(FL_overallMCC_resolutions, 1)), 1));
        figure('position', [50, 50, 1000, 500])
        plot(meanMCC_firstLayer_resolutions, 'r', 'LineWidth', 2); hold on;
        plot(meanMCC_firstLayer_resolutions + stdMCC_firstLayer_resolutions, '--r', 'LineWidth', 1); hold on;
        plot(meanMCC_firstLayer_resolutions - stdMCC_firstLayer_resolutions, '--r', 'LineWidth', 1);
        xlabel('B-Spline Widths'); ylabel('MCCs'); 
%         set(gca,'xtick',[1:0.2:5, 6:26], 'xticklabels', [5:25, 50:5:150])
        set(gca, 'FontName', 'Arial','FontWeight','bold', 'FontSize', 24)
        meanMCC_firstLayer = squeeze(mean(FL_overallMCC, 1));
        stdMCC_firstLayer = squeeze(std(FL_overallMCC, 0, 1));
        
        disp('MCC of base learners:')
        disp(meanMCC_firstLayer_resolutions)
        
        % Show performance - Highest mean(MCC)/STD
        tempResult = meanMCC_firstLayer ./ stdMCC_firstLayer;
        index1 = find(tempResult == max(max(tempResult)));
        if length(index1) > 1
            index1 = index1(1);
        end
        bestMCC1 = meanMCC_firstLayer(index1);
        bestSTD1 = stdMCC_firstLayer(index1);
        disp(['The best NestedCV MDM MCC: MCC=', mat2str(bestMCC1), ' (STD=', mat2str(bestSTD1), ')']);
        
        
        %% Second Layer Performance
        SL_overallMCC = zeros(repeatSize, numFold, 1);
        for repeat = 1:repeatSize
            tempPredictions_secondLayer = squeeze(overallPredictions_secondLayer(repeat, :, :));
            % For each model from folds
            for fold = 1:size(tempPredictions_secondLayer)
                tempModelPrediction_secondLayer = tempPredictions_secondLayer(fold, :);
                
                CM_fold_secondLayer = confusionmat(tempModelPrediction_secondLayer, overallTarget);
                if (size(CM_fold_secondLayer,1)==1&&size(CM_fold_secondLayer,2)==1)
                    CM_fold_secondLayer = [CM_fold_secondLayer(1,1) 0;0 0];
                end
                MCC_fold_secondLayer = mcc(CM_fold_secondLayer);
                SL_overallMCC(repeat, fold) = MCC_fold_secondLayer;
            end
        end
        
        % Concatenent the Repeat * Fold matrix into long vector
        SL_overall_vector = SL_overallMCC(:);
        meanMCC_secondLayer = max(SL_overall_vector);
        stdMCC_secondLayer = std(SL_overall_vector, 0, 1);
        bestMCC2 = meanMCC_secondLayer;
        bestSTD2 = stdMCC_secondLayer;
        
        %% Table Visualization
        NestedCVMDM_BestMCC = bestMCC1;
        if length(NestedCVMDM_BestMCC) > 1
            NestedCVMDM_BestMCC = NestedCVMDM_BestMCC(1);
        end
        NestedCVMDM_BestSTD = bestSTD1;
        if length(NestedCVMDM_BestSTD) > 1
            NestedCVMDM_BestSTD = NestedCVMDM_BestSTD(1);
        end
        NestedCVDLMDM_BestMCC = bestMCC2;
        if length(NestedCVDLMDM_BestMCC) > 1
            NestedCVDLMDM_BestMCC = NestedCVDLMDM_BestMCC(1);
        end
        NestedCVDLMDM_BestSTD = bestSTD2;
        if length(NestedCVDLMDM_BestSTD) > 1
            NestedCVDLMDM_BestSTD = NestedCVDLMDM_BestSTD(1);
        end
        
        disp(['The best NestedCV DLMDM MCC: MCC=', mat2str(NestedCVDLMDM_BestMCC), ' (STD=', mat2str(NestedCVDLMDM_BestSTD), ')']);
        
        titleStr = category;
        x=[NestedCVMDM_BestMCC NestedCVDLMDM_BestMCC]';
        y=[NestedCVMDM_BestSTD NestedCVDLMDM_BestSTD]';
      
        %% Save Results
%         save(['Results\',Subject, '&', mat2str(nCategories),'Categories\Summarized_NestedCVDLMDM_',Subject, '&', mat2str(nCategories),'Categories_', category,'_',runCase,'_minRes.mat'], 'NestedCVMDM_BestMCC', 'NestedCVDLMDM_BestMCC', 'NestedCVMDM_BestSTD', 'NestedCVDLMDM_BestSTD', 'FL_overallMCC', 'SL_overallMCC');
%         close figure 1
    end
end