%% This script is used to generate the two-neuron synthetic spatio-temporal patterns
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-6-23

clear;clc;

%% Simulation parameters - User entering required
dt = 1/500; % Step size, unit: ms
tSim = 4; % how many seconds we are going to simulate
fr_base = 5; % Baseline firing rate, unit: Hz

L = tSim/dt; % length of pattern
t =[1:1:L]'*dt; % time

nTrials = 100; % number of simulated trials (model instances)
nNeuron = 2;
runCase = 'CN&CS'; % Combined Neuron + Combined Resolution

% Nunber of categories we are going to simulate
nCategories = 2;

%% Simulated Neurons - Combined neuron case
neuronPIFs_combined = zeros(nNeuron, nCategories, L); % Neuron * Categories * DecodingWindow
neuronSpike_combined = zeros(nNeuron, nCategories, L, nTrials); % Neuron * Categories * DecodingWindow * Trials
spikeTensor_combined = zeros(nNeuron*nTrials, L, nNeuron);

% Low-resolution neuron - you can replace with new designed patterns
load('Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_SingleNeuron_2Categories_SN&LS4.mat')
neuronPIF1 = neuronPIFs;
neuronSpike1 = neuronSpikes;
spikeTensor1 = spikeTensor;

% High-resolution neuron - you can replace with new designed patterns
load('Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_SingleNeuron_2Categories_SN&HS4.mat')
neuronPIF2 = neuronPIFs;
neuronSpike2 = neuronSpikes;
spikeTensor2 = spikeTensor;

% Combined neurons
neuronPIFs_combined(1, :, :) = neuronPIF1;
neuronPIFs_combined(2, :, :) = neuronPIF2;
neuronSpike_combined(1, :, :, :) = neuronSpike1;
neuronSpike_combined(2, :, :, :) = neuronSpike2;
spikeTensor_combined(:, :, 1) = spikeTensor1;
spikeTensor_combined(:, :, 2) = spikeTensor2;

neuronPIFs = neuronPIFs_combined;
neuronSpikes = neuronSpike_combined;
spikeTensor = spikeTensor_combined;

% Note by Xiwei: backup previous data/results before run this
% Be care of overwritting
save(['Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_CombinedNeuron_', mat2str(nCategories),'Categories_', runCase,'.mat'], 'spikeTensor', 'categoriesTarget', 'neuronSpikes', 'randomSeeds', 'neuronPIFs', 'muPool', 'sigmaPool', 'intensePool')


%% Visualization : Show patterns - raster & PIF plot
% Exp patterns
for c = 1
    figure('Position',[50 0 500 500]);
    for i = 1:nNeuron
        subplot(2, 1, i)
        p = squeeze(squeeze(neuronPIFs(i, c, :)));
        line(t,p, 'Color', 'r', 'lineWidth', 2); ylabel('PIFs');
        ax1 = gca;
        ax1.XColor = 'r'; ax1.YColor = 'r';
        ax1_pos = ax1.Position;
        ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right', 'Color', 'none');
        x = squeeze(squeeze(neuronSpikes(i, c, :, :)));
        rasterplot(x'); ylabel(['Neuron #', mat2str(i)]);xlabel('time');
        ax2.XColor = 'k'; ax2.YColor = 'k';
    end
    
end

% Control patterns
figure('Position',[50 0 500 500]);
for c=2
    for i = 1:nNeuron
        subplot(2, 1, i)
        p = squeeze(squeeze(neuronPIFs(i, c, :)));
        line(t,p, 'Color', 'r', 'lineWidth', 2); ylabel('PIFs');
        ax1 = gca;
        ax1.XColor = 'r'; ax1.YColor = 'r';
        ax1_pos = ax1.Position;
        ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right', 'Color', 'none');
        x = squeeze(squeeze(neuronSpikes(i, c, :, :)));
        rasterplot(x'); ylabel(['Neuron #', mat2str(i)]);xlabel('time');
        ax2.XColor = 'k'; ax2.YColor = 'k';
    end
end