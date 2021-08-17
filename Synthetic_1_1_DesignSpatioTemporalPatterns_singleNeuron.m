%% This script is used to generate single-neuron synthetic spatio-temporal patterns
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-06-23

clear;clc;

%% Uncomment/comment corresponding cases you want to check

% Case 1: single neuron + low resolution
muPool = 1;
sigmaPool = 0.3;
intensePool = 0.02;
nTrials = 100; % number of simulated trials (model instances)
runCase = 'SN&LS'; % SingleNueron + HighResolution + LowIntensity

% % Case 2: single neuron + high resolution
% muPool = 3;
% sigmaPool = 0.003;
% intensePool = 0.005;
% nTrials = 100; % number of simulated trials (model instances)
% runCase = 'SN&HS'; % SingleNueron + HighResolution + LowIntensity

%% Simulation parameters - User entering required
dt = 1/500; % Step size, unit: ms
tSim = 4; % how many seconds we are going to simulate
fr_base = 5; % Baseline firing rate, unit: Hz

L = tSim/dt; % length of pattern
t =[1:1:L]'*dt; % time

% Nunber of categories we are going to simulate
nCategories = 2;

%% Start genenrating spikes

neuronSpikes = zeros(nCategories, L, nTrials);
neuronPIFs = zeros(nCategories, L);
randomSeeds = zeros(nCategories, 3);
% Baseline
p0 = ones(L,1)*fr_base*dt; % prob. intensity function (# of spikes per bin);

%% Simulated Neurons - single neuron case
% Exp neuron
[neuronSpikes(1, :, :), neuronPIFs(1, :)] = GenerateSimulatedNeuron_SN_exp(muPool, sigmaPool, p0, nTrials, t, L, intensePool);
% A*(p1+p0) + (B-A)*p0 = B*p_new -> p_new = A/B*p1 + p0
totalArea = normcdf(t, muPool, sigmaPool);
A = (totalArea(end) - totalArea(1)) * intensePool;
B = 4 * max(neuronPIFs(1, :));
p0_ctrl = A/B * max(neuronPIFs(1, :)) + p0;

% Control neuron - same averaged firing rate
[neuronSpikes(2, :, :), neuronPIFs(2, :)] = GenerateSimulatedNeuron_SN_ctrl(muPool, sigmaPool, p0_ctrl, nTrials, t, L, intensePool);

%% Start genenrating spikes tensor for DLMDM
spikeTensor = zeros(nTrials*nCategories, tSim / dt); % Instrance * timeWindow

categoriesTarget = zeros(nCategories, nTrials*nCategories);

tempSeq = 1:nTrials*nCategories;
tempTrial = mod(tempSeq, nCategories);
indexC = ones(nCategories, 1);
for i = 1:nTrials*nCategories
    if tempTrial(i) == 0
        categoriesTarget(1, i) = 1;
        tempTensor = squeeze(neuronSpikes(1, :, :));
        spikeTensor(i, :, :) = tempTensor(:, indexC(1))';
        indexC(1) = indexC(1) + 1;
    elseif tempTrial(i) == 1
        categoriesTarget(2, i) = 1;
        tempTensor = squeeze(neuronSpikes(2, :, :));
        spikeTensor(i, :) = tempTensor(:, indexC(2))';
        indexC(2) = indexC(2) + 1;
    elseif tempTrial(i) == 2
        categoriesTarget(3, i) = 1;
        tempTensor = squeeze(neuronSpikes(3, :, :));
        spikeTensor(i, :, :) = tempTensor(:, indexC(3))';
        indexC(3) = indexC(3) + 1;
    elseif tempTrial(i) == 3
        categoriesTarget(4, i) = 1;
        tempTensor = squeeze(neuronSpikes(4, :, :));
        spikeTensor(i, :, :) = tempTensor(:, indexC(4))';
        indexC(4) = indexC(4) + 1;
    elseif tempTrial(i) == 4
        categoriesTarget(5, i) = 1;
        tempTensor = squeeze(neuronSpikes(5, :, :));
        spikeTensor(i, :, :) = tempTensor(:, indexC(5))';
        indexC(5) = indexC(5) + 1;
    else
        disp("Error in category definition!")
    end
end

% Note by Xiwei: backup previous data/results before run this
% Be care of overwritting
save(['Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_SingleNeuron_', mat2str(nCategories),'Categories_', runCase,'.mat'], 'spikeTensor', 'categoriesTarget', 'neuronSpikes', 'randomSeeds', 'neuronPIFs', 'muPool', 'sigmaPool', 'intensePool')


%% Visualization : Show patterns - raster & PIF plot
% Exp patterns
for c = 1
    figure('Position',[50 0 500 500]);
    p = squeeze(squeeze(neuronPIFs(c, :)));
    line(t,p, 'Color', 'r', 'lineWidth', 2); ylabel('PIFs');
    ax1 = gca;
    ax1.XColor = 'r'; ax1.YColor = 'r';
    ax1_pos = ax1.Position;
    ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right', 'Color', 'none');
    x = squeeze(squeeze(neuronSpikes(c, :, :)));
    rasterplot(x'); ylabel(['Neuron']);xlabel('time');
    ax2.XColor = 'k'; ax2.YColor = 'k';
    
end

% Control patterns
figure('Position',[50 0 500 500]);
for c=2
    p = squeeze(squeeze(neuronPIFs(c, :)));
    line(t,p, 'Color', 'r', 'lineWidth', 2); ylabel('PIFs');
    ylim([0 0.04])
    ax1 = gca;
    ax1.XColor = 'r'; ax1.YColor = 'r';
    ax1_pos = ax1.Position;
    ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right', 'Color', 'none');
    x = squeeze(squeeze(neuronSpikes(c, :, :)));
    rasterplot(x'); ylabel(['Neuron']);xlabel('time');
    ax2.XColor = 'k'; ax2.YColor = 'k';
end