%% This script is used to generate input data for Memory Decoding model 
% synthetic two neuron case
% It generates 10 output files for 10-fold nested cross-validation
% Program: RAM USC
% Author: Xiwei She
% Date: 2020-06-23

clear;clc

%% Load data
% Case 3: combine case 1 & 2 CN&CS2 can be used
nCategories = 2;
runCase = 'CN&CS4';
load(['Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_CombinedNeuron_', mat2str(nCategories),'Categories_', runCase,'.mat'])
for ca = 1:nCategories
    switch ca
        case 1
            category = 'A';
        case 2
            category = 'B';
        case 3
            category = 'C';
        case 4
            category = 'D';
        case 5
            category = 'E';
    end
    target = categoriesTarget(ca, :)';
    SpikeTensor = spikeTensor;

    %% Make nested cv partitioning
    partitionFolds = 10; % 10
    CrossValSet = cvpartition(length(target),'KFold', partitionFolds);

    for partition = 1:partitionFolds
        TrainingSet_SpikeTensor = SpikeTensor(training(CrossValSet, partition),:, :);
        TrainingSet_target = target(training(CrossValSet, partition),:, :);
        TestingSet_SpikeTensor = SpikeTensor(test(CrossValSet, partition),:, :);
        TestingSet_target = target(test(CrossValSet, partition),:, :);

        % Save Output File - Be care of overwritting
        oF = strcat('Synthetic_Input\CombinedNeuron&2Categories\CombinedNeuron&', mat2str(nCategories), 'Categories','\NestCVDLMDM_synthetic_CombinedNeuron&', mat2str(nCategories),'Categories_', category, '_fold_', mat2str(partition),'_', runCase,'.mat');
        save(oF, 'TrainingSet_SpikeTensor', 'TrainingSet_target', 'TestingSet_SpikeTensor', 'TestingSet_target', 'target', 'SpikeTensor', 'CrossValSet')
    end
end