%% This script is used for checking the SCFM of population neuron simulation cases
% Corresponding to section 3.1 (Fig.8) in the manuscript of NECO-2021-She
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-11-26
clear; clc;
addpath(genpath('../toolbox'));

% Case 4: realistic
nCategories = 5;
nNeuron = 30; globalIndex = 4;
nTrials = 500;  runCase = [mat2str(nNeuron), 'N&', mat2str(nTrials), 'T&', mat2str(nCategories),'C']; 

category = 'A'; % As an example, can change to A, B, C, D, or E

%% load data and results

iF1 = strcat('Results\PopulationNeurons&5Categories\NestedCVDLMDM_synthetic_Realistic&_',runCase,'_Category', category,'.mat');
load(iF1);

iF3 = strcat('Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_Realistic_',runCase,'.mat');
load(iF3, 'spikeTensor', 'neuronSpikes', 'neuronPIFs');
SpikeTensor = spikeTensor;
%% Get variables
numNeuron = nNeuron;
resolutionPool = MDfit.m_all;
L = MDfit.L; % Length of decoding window 2000
order = MDfit.d; % Order 3
numFold1 = length(MDfit.R_first(1, 1).FL_inside_Coefficients);

FMask_first_matrix_folds = zeros(L+1, numNeuron, numFold1);
BSpline_first_fold_neurons = zeros(L+1, 154, numFold1, nNeuron);
probFMask_first_matrix = zeros(length(resolutionPool), L+1, numNeuron);
probC0_first_matrix = zeros(length(resolutionPool), 1);
Bspline_first_neurons = zeros(length(resolutionPool), L+1, 154, nNeuron);
Bspline_first_w0_folds = zeros(numFold1, 1);
Bspline_first_w0 = zeros(length(resolutionPool), 1);
%% First layer SCFM

for resolution = resolutionPool
    resolutionIndex = find(resolutionPool == resolution);
    
    J = resolution+order+1; % BSpline knots for current resolution
    
    for fold = 1:numFold1
        w0_firstLayerPool = MDfit.R_first(globalIndex, resolutionIndex).FL_outside_GlobalC0;
        weights_firstLayerPool = MDfit.R_first(globalIndex, resolutionIndex).FL_outside_GlobalCoefficients;
        
        w0_firstLayer = w0_firstLayerPool(fold);
        weights_firstLayer = weights_firstLayerPool(fold, :);
        
        % B-Spline tools - Eq. 8&9
        BSpline = bspline(order+1, resolution+2, L+1);
        
        % Reshape the weights
        weights2_firstLayer = reshape(weights_firstLayer, J, []);
        weights2_firstLayer(weights2_firstLayer<0) = 0; % Test
        
        % Get the Functional Matrics - Eq. 15
        F_maskTemp = BSpline * weights2_firstLayer; % F is always with the weights
        
        % Fold averaged the F
        FMask_first_matrix_folds(:, :, fold) = F_maskTemp;
        
        % For bspline visualization
        for n = 1:nNeuron
            BSpline_first_fold_neurons(:, 1:size(BSpline, 2), fold, n) = 1 ./ (1+ exp(-BSpline .* weights2_firstLayer(:, n)' - w0_firstLayer/1.5) );
            BSpline_first_fold_neurons(:, size(BSpline, 2)+1:end, fold, n) = 1 ./ (1+ exp(-w0_firstLayer/1.5) );
        end
        Bspline_first_w0_folds(fold) = 1 ./ (1 + exp(-w0_firstLayer/1.5));
        
    end
    Bspline_first_w0(resolutionIndex, 1) = mean(Bspline_first_w0_folds);
    
    % Calculate the probability matrices - Multi Trial Multi Resolution
    probTemp = mean(FMask_first_matrix_folds, 3); % Un-weighted B-spline
    probTemp_neurons = squeeze(mean(BSpline_first_fold_neurons, 3));
    Bspline_first_neurons(resolutionIndex, :, :, :) = probTemp_neurons;
    w0Temp = mean(w0_firstLayerPool);
    probFMask_first_matrix(resolutionIndex, :, :) = 1 ./ (1+ exp(-w0Temp - probTemp) );
    probC0_first_matrix(resolutionIndex) = 1 ./ (1+ exp(-w0Temp) );
end

SCFM_firstLayer_0 = probFMask_first_matrix;
SCFM_firstLayer = squeeze(mean(SCFM_firstLayer_0, 1));
SCFM_firstLayer_w0 = mean(probC0_first_matrix);

% First layer SCFM
% figure('Position', [50, 50, 900, 900]);
% SCFM_firstLayer_vis = SCFM_firstLayer - SCFM_firstLayer_w0;
% CustomColorMap(1, -1)
% pcolorfull(SCFM_firstLayer_vis')
% colorbar;

%% Second Layer SCFM
SL_inside_Coefficients = MDfit.R_second(globalIndex).SL_inside_Coefficients;
SL_inside_FitInfo = MDfit.R_second(globalIndex).SL_inside_FitInfo;
SL_inside_Deviance = MDfit.R_second(globalIndex).SL_inside_Deviance;

globalMinDeviance2 = min(SL_inside_Deviance);
globalIndex2 = find(SL_inside_Deviance == globalMinDeviance2);
if length(globalIndex2) > 1
    globalIndex2 = globalIndex2(1);
end

numFold = length(SL_inside_Coefficients);
SL_outside_GlobalCoefficients = zeros(numFold, size(SL_inside_Coefficients{1}, 1));
SL_outside_GlobalC0 = zeros(numFold, 1);
for tempI = 1:numFold
    globalC0 = SL_inside_FitInfo{tempI}.Intercept(globalIndex2);
    SL_outside_GlobalC0(tempI) = globalC0;
    SL_outside_GlobalCoefficients(tempI, :) = SL_inside_Coefficients{tempI}(:, globalIndex2);
end

% Second Layer F & Coef
probFMask_first_permute = SCFM_firstLayer_0(:, :); % Resolution * [decodingWindow * Neuron]
FMask_second_prob_matrix = zeros(numFold, L+1, numNeuron);
FMask_second_prob_matrix_baseline = zeros(numFold, 1);
for f = 1:numFold
    w0Temp = SL_outside_GlobalC0(f);
    weightsTemp = SL_outside_GlobalCoefficients(f, :);
    FMask_second_0 = weightsTemp * probFMask_first_permute;
    FMask_second_0_firstLayer =  weightsTemp * probC0_first_matrix; % Baseline of first-layer
    FMask_second = reshape(FMask_second_0, L+1, []); % DecodingWindow * Neuron

    FMask_second_prob_matrix(f, :, :) = 1 ./ (1 + exp(-w0Temp-FMask_second) ); 
    FMask_second_prob_matrix_baseline(f, :) = 1 ./ (1 + exp(-w0Temp-FMask_second_0_firstLayer) ); % Consider baseline of the first-layer
end

% F to Prob for this trial by averaged the fold prob
probFMask_second_matrix = squeeze(mean(FMask_second_prob_matrix, 1));
probFMask_second_matrix_baseline = squeeze(mean(FMask_second_prob_matrix_baseline));

% Don't consider first-layer baseline
SCFM_Prob_final = probFMask_second_matrix';
SCFM_Prob_final_vis = SCFM_Prob_final - probFMask_second_matrix_baseline;
SCFM_Prob_final_vis(SCFM_Prob_final_vis<0) = 0; % Now only consider positive patterns

%% Now create SCFM Mask - POSITIVE
maskThreshold = mean(mean(SCFM_Prob_final_vis));
SCFM_Mask = ( (SCFM_Prob_final_vis - abs(maskThreshold/5))  > 0 );
SCFM_Mask = SCFM_Mask(:, 1:2000);

%% SCFM Visualization
figure('Position', [50, 50, 900, 800]);
CustomColorMap(1, -1)
pcolorfull(SCFM_Prob_final_vis); % colorbar
tempRange = max(-min(SCFM_Prob_final_vis(:)), max(SCFM_Prob_final_vis(:)));
caxis([-tempRange, tempRange])
%xlabel('Time (-2 to 2s)'); 
%ylabel('Neurons');
title(['SCFM (', category, ')']);
set(gca, 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 42)

%% Spatio-temporal patterns
SP11 = squeeze(squeeze(neuronSpikes(:, 1, :, 1))); % The first class
SP1 = squeeze(squeeze(mean(neuronSpikes(:, 1, :, :), 4))); % The first class
SP1_std = std(sum(squeeze(sum(neuronSpikes(:, 1, :, :), 1)), 1)); % The first class
SP21 = squeeze(squeeze(neuronSpikes(:, 2, :, 1))); % The first class
SP2 = squeeze(squeeze(mean(neuronSpikes(:, 2, :, :), 4))); % The second class
SP2_std = std(sum(squeeze(sum(neuronSpikes(:, 2, :, :), 1)), 1)); % The second class
SP31 = squeeze(squeeze(neuronSpikes(:, 3, :, 1))); % The first class
SP3 = squeeze(squeeze(mean(neuronSpikes(:, 3, :, :), 4))); % The third class
SP3_std = std(sum(squeeze(sum(neuronSpikes(:, 3, :, :), 1)), 1)); % The third class
SP41 = squeeze(squeeze(neuronSpikes(:, 4, :, 1))); % The first class
SP4 = squeeze(squeeze(mean(neuronSpikes(:, 4, :, :), 4))); % The forth class
SP4_std = std(sum(squeeze(sum(neuronSpikes(:, 4, :, :), 1)), 1)); % The forth class
SP51 = squeeze(squeeze(neuronSpikes(:, 5, :, 1))); % The first class
SP5 = squeeze(squeeze(mean(neuronSpikes(:, 5, :, :), 4))); % The fifth class
SP5_std = std(sum(squeeze(sum(neuronSpikes(:, 5, :, :), 1)), 1)); % The fifth class

%% Spike count Classification
% Masked spatio-temporal patterns
SP1_masked = SP1 .* SCFM_Mask;
SP2_masked = SP2 .* SCFM_Mask;
SP3_masked = SP3 .* SCFM_Mask;
SP4_masked = SP4 .* SCFM_Mask;
SP5_masked = SP5 .* SCFM_Mask;

temp = squeeze(neuronSpikes(:, 1, :, :)) .* SCFM_Mask;
SP1_std_masked = std(squeeze(sum(sum(temp))));
temp = squeeze(neuronSpikes(:, 2, :, :)) .* SCFM_Mask;
SP2_std_masked = std(squeeze(sum(sum(temp))));
temp = squeeze(neuronSpikes(:, 3, :, :)) .* SCFM_Mask;
SP3_std_masked = std(squeeze(sum(sum(temp))));
temp = squeeze(neuronSpikes(:, 4, :, :)) .* SCFM_Mask;
SP4_std_masked = std(squeeze(sum(sum(temp))));
temp = squeeze(neuronSpikes(:, 5, :, :)) .* SCFM_Mask;
SP5_std_masked = std(squeeze(sum(sum(temp))));

SP1_bar0 = sum(SP1, 2);
SP2_bar0 = sum(SP2, 2);
SP3_bar0 = sum(SP3, 2);
SP4_bar0 = sum(SP4, 2);
SP5_bar0 = sum(SP5, 2);
SP1_bar1 = sum(SP1_masked, 2);
SP2_bar1 = sum(SP2_masked, 2);
SP3_bar1 = sum(SP3_masked, 2);
SP4_bar1 = sum(SP4_masked, 2);
SP5_bar1 = sum(SP5_masked, 2);

figure('Position', [50, 50, 900, 800]);
subplot(2, 1, 1)
range1 = max([sum(SP1_bar0), sum(SP2_bar0), sum(SP3_bar0), sum(SP4_bar0), sum(SP5_bar0)]);
barwitherr([SP1_std, SP2_std, SP3_std, SP4_std, SP5_std], [sum(SP1_bar0), sum(SP2_bar0), sum(SP3_bar0), sum(SP4_bar0), sum(SP5_bar0)]);ylim([0, range1+50])
%ylabel('# of spikes')
%title('# of Spikes (without SCFM)')
set(gca, 'xticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 42)
subplot(2, 1, 2)
range2 = max([sum(SP1_bar1), sum(SP2_bar1), sum(SP3_bar1), sum(SP4_bar1), sum(SP5_bar1)]);
barwitherr([SP1_std_masked, SP2_std_masked, SP3_std_masked, SP4_std_masked, SP5_std_masked], [sum(SP1_bar1), sum(SP2_bar1), sum(SP3_bar1), sum(SP4_bar1), sum(SP5_bar1)], 'r'); ylim([0, range2+20])
%ylabel('# of spikes'); 
%xlabel('Decoding Categories');
%title(['# of Spikes (with SCFM-', category, ')'])
set(gca, 'xticklabel', ['A'; 'B'; 'C'; 'D'; 'E'], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 42)