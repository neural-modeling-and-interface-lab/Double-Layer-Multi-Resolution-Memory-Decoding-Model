%% This script is used for checking the SCFM of two-neuron simulation cases
% Corresponding to section 3.1 (Fig.7) in the manuscript of NECO-2021-She
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-11-26
clear; clc;
addpath(genpath('../toolbox'));

% Case 3: combine case 1 & 2
runCase = 'CN&CS4';
Subject = 'CombinedNeuron';
nCategories = 2;
nNeuron = 2;
globalIndex = 8;
plotTitle = 'Model Weighted B-Spline Fitting';
plotTitle2 = 'Model Filtered B-Spline Fitting';

Category = 'A';


%% load data and results
iF1 = strcat('Results\',Subject, '&', mat2str(nCategories),'Categories\NestedCVDLMDM_synthetic_',Subject, '&', mat2str(nCategories),'Categories_', Category,'_', runCase,'_globalSelected.mat');
load(iF1);

iF3 = strcat('Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_',Subject, '_', mat2str(nCategories),'Categories_', runCase,'.mat');
load(iF3, 'spikeTensor', 'neuronSpikes', 'neuronPIFs');
SpikeTensor = spikeTensor;

%% Get variables
numNeuron = nNeuron;
resolutionPool = MDfit.m_all;
L = MDfit.L; % Length of decoding window 2000
order = MDfit.d; % Order 3
numFold1 = length(MDfit.R_first(1, 1).FL_inside_Coefficients);

FMask_first_matrix_folds = zeros(L+1, numNeuron, numFold1);
BSpline_first_fold = zeros(L+1, 154, numFold1);
BSpline_first_fold2 = zeros(L+1, 154, numFold1);
probFMask_first_matrix_folds = zeros(L+1, numNeuron, numFold1);
probFMask_first_matrix = zeros(length(resolutionPool), L+1, numNeuron);
probC0_first = zeros(length(resolutionPool), 1);
C0_first = zeros(length(resolutionPool), 1);
Bspline_first = zeros(length(resolutionPool), L+1, 154);
Bspline_first2 = zeros(length(resolutionPool), L+1, 154);
Bspline_first_w0_folds = zeros(numFold1, 1);
Bspline_first_w0 = zeros(length(resolutionPool), 1);
%% First layer SCFM
for resolution = resolutionPool
    resolutionIndex = find(resolutionPool == resolution);
    
    J = resolution+order+1; % BSpline knots for current resolution
    
    for fold = 1:numFold1
%     for fold = 1:1
        w0_firstLayerPool = MDfit.R_first(globalIndex, resolutionIndex).FL_outside_GlobalC0;
        weights_firstLayerPool = MDfit.R_first(globalIndex, resolutionIndex).FL_outside_GlobalCoefficients;
        
        w0_firstLayer = w0_firstLayerPool(fold);
        weights_firstLayer = weights_firstLayerPool(fold, :);
        
        % B-Spline tools - Eq. 8&9
        BSpline = bspline(order+1, resolution+2, L+1);
        
        % Reshape the weights
        weights2_firstLayer = reshape(weights_firstLayer, J, []);
        weights2_firstLayer(weights2_firstLayer<0) = 0; % Here we only consider positive patterns
        
        % Get the Functional Matrics - Eq. 15
        F_maskTemp = BSpline * weights2_firstLayer; % F is always with the weights
        
        % Fold averaged the F
        FMask_first_matrix_folds(:, :, fold) = F_maskTemp;
        
        % For bspline visualization
        BSpline_first_fold(:, 1:size(BSpline, 2), fold) = 1 ./ (1+ exp(-BSpline .* weights2_firstLayer(:, 1)' - w0_firstLayer) );
        BSpline_first_fold(:, size(BSpline, 2)+1:end, fold) = 1 ./ (1+ exp(-w0_firstLayer) );
        BSpline_first_fold2(:, 1:size(BSpline, 2), fold) = 1 ./ (1+ exp(-BSpline .* weights2_firstLayer(:, 2)' - w0_firstLayer) );
        BSpline_first_fold2(:, size(BSpline, 2)+1:end, fold) = 1 ./ (1+ exp(-w0_firstLayer) );
        Bspline_first_w0_folds(fold) = 1 ./ (1 + exp(-w0_firstLayer));
    end
    % Calculate the probability matrices - Multi Trial Multi Resolution
    probTemp = mean(FMask_first_matrix_folds, 3); % Un-weighted B-spline
    probTemp2 = mean(BSpline_first_fold, 3); % Un-weighted B-spline
    probTemp3 = mean(BSpline_first_fold2, 3); % Un-weighted B-spline
    Bspline_first(resolutionIndex, :, :) = probTemp2;
    Bspline_first2(resolutionIndex, :, :) = probTemp3;
    Bspline_first_w0(resolutionIndex, 1) = mean(Bspline_first_w0_folds);
    
    w0Temp = mean(w0_firstLayerPool);
    probFMask_first_matrix(resolutionIndex, :, :) = 1 ./ (1+ exp(-w0Temp - probTemp) );
    probC0_first(resolutionIndex, 1) = 1 ./ (1+ exp(-w0Temp) );
    C0_first(resolutionIndex, 1) = w0Temp;
end
% Ensemble first-layer SCFM
SCFM_firstLayer_0 = probFMask_first_matrix;
SCFM_firstLayer = mean(SCFM_firstLayer_0, 1);

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
FMask_second_prob_matrix_lowRes = zeros(numFold, L+1, numNeuron);
FMask_second_prob_matrix_highRes = zeros(numFold, L+1, numNeuron);
C0_second_prob_matrix = zeros(numFold, L+1, numNeuron);
for f = 1:numFold
    w0Temp = SL_outside_GlobalC0(f);
    weightsTemp = SL_outside_GlobalCoefficients(f, :);
    
    FMask_second_0 = w0Temp + probFMask_first_permute' * weightsTemp';% Need w0 here? - yes
    
    FMask_second_lowRes = w0Temp + probFMask_first_permute(1:21, :)' * weightsTemp(1:21)';
    FMask_second_highRes = w0Temp + probFMask_first_permute(22:42, :)' * weightsTemp(22:42)';
    
    FMask_second = reshape(FMask_second_0, L+1, []); % DecodingWindow * Neuron
    FMask_second_lowRes = reshape(FMask_second_lowRes, L+1, []);
    FMask_second_highRes = reshape(FMask_second_highRes, L+1, []);
    
    FMask_second_prob_matrix(f, :, :) = 1 ./ (1 + exp(-FMask_second) ); % F to Prob for this Fold
    FMask_second_prob_matrix_lowRes(f, :, :) = 1 ./ (1 + exp(-FMask_second_lowRes) );
    FMask_second_prob_matrix_highRes(f, :, :) = 1 ./ (1 + exp(-FMask_second_highRes) );
end

% F to Prob for this trial by averaged the fold prob
probFMask_second_matrix = squeeze(mean(FMask_second_prob_matrix, 1));
probFMask_second_matrix_lowRes = squeeze(mean(FMask_second_prob_matrix_lowRes, 1));
probFMask_second_matrix_highRes = squeeze(mean(FMask_second_prob_matrix_highRes, 1));

SL_weights = mean(SL_outside_GlobalCoefficients, 1);
SL_C0 = mean(SL_outside_GlobalC0, 1);
FMask_second_matix = zeros(length(resolutionPool), size(Bspline_first, 2), size(Bspline_first, 3));
FMask_second_matix2 = zeros(length(resolutionPool), size(Bspline_first, 2), size(Bspline_first, 3));
FMask_second_matix_unweighted = zeros(length(resolutionPool), size(Bspline_first, 2), size(Bspline_first, 3));
FMask_second_matix2_unweighted = zeros(length(resolutionPool), size(Bspline_first, 2), size(Bspline_first, 3));
for res = 1:length(resolutionPool)
    FMask_first_matrix = squeeze(Bspline_first(res, :, :));
    FMask_first_matrix2 = squeeze(Bspline_first2(res, :, :));
    
    % Consider intecept from all other classifier
    weights_temp = SL_weights; 
    weights_temp(res) = [];
    probC0_first_temp = Bspline_first_w0;
    probC0_first_temp(res) = [];

    inteceptTemp = weights_temp * probC0_first_temp;
    FMask_second_matix(res, :, :) = 1 ./ (1 + exp( -FMask_first_matrix .* SL_weights(res) - SL_C0 - inteceptTemp) );
    FMask_second_matix2(res, :, :) = 1 ./ (1 + exp( -FMask_first_matrix2 .* SL_weights(res) - SL_C0 - inteceptTemp) );
end
% SCFM
SCFM_Prob_final = probFMask_second_matrix;
SCFM_Prob_final_vis = SCFM_Prob_final;
SCFM_Prob_final_vis(SCFM_Prob_final_vis<0) = 0; % Here we only consider positive patterns

%% Start visual
% SCFM & BSplines
yRange_1 = min( min(min(SCFM_Prob_final_vis(:, 1))), min(min(min(FMask_second_matix))) );
yRange_2 = max( max(max(SCFM_Prob_final_vis(:, 1))), max(max(max(FMask_second_matix))) );
offsetSCFM = min(min(SCFM_Prob_final_vis(:, 1))) - min(min(min(FMask_second_matix)));
figure('Position',[50 50 1400 700]);
hSCFM = area(SCFM_Prob_final_vis(:, 1)'-offsetSCFM);
hSCFM.FaceColor = [0.68, 0.68, 0.68];
hSCFM.LineStyle = 'none';
% ylabel('SCFM')
xlim([0, 2000])
ylim([max(yRange_1-0.05, 0), yRange_2+0.05])
set(gca, 'xtick', [])
set(gca, 'box', 'off', 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 22,'YAxisLocation','right')
ax1 = gca;
ax1.XColor = [0.5, 0.5, 0.5]; ax1.YColor = [0.5, 0.5, 0.5];
ax1_pos = ax1.Position;
ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right', 'Color', 'none');
cmap = jet(sum(SL_weights~=0));
cIndex = 0;
for res = 1:size(SCFM_firstLayer_0, 1) % B-splines
    if SL_weights(res) > 0
        cIndex = cIndex + 1;
        plotTemp = squeeze(FMask_second_matix(res, :, :));
        plot(plotTemp, 'color', cmap(cIndex, :), 'lineWidth', 2); hold on;
        xlim([0, 2000]); ylim([max(yRange_1-0.05, 0), yRange_2+0.05])
        set(gca, 'box', 'off')
        set(gca, 'Color', 'none')
    end
end
xticks([0, 250 ,500, 750, 1000, 1250, 1500, 1750, 2000])
xticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
% xlabel('Decoding Window (s)');
set(gca, 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 22)

yRange_1 = min( min(min(SCFM_Prob_final_vis(:, 2))), min(min(min(FMask_second_matix2))) );
yRange_2 = max( max(max(SCFM_Prob_final_vis(:, 2))), max(max(max(FMask_second_matix2))) );
offsetSCFM = min(min(SCFM_Prob_final_vis(:, 2))) - min(min(min(FMask_second_matix2)));
figure('Position',[50 50 1400 700]);
hSCFM = area(SCFM_Prob_final_vis(:, 2)'-offsetSCFM);
hSCFM.FaceColor = [0.68, 0.68, 0.68];
hSCFM.LineStyle = 'none';
% ylabel('SCFM')
xlim([0, 2000])
ylim([max(yRange_1-0.05, 0), yRange_2+0.05])
set(gca, 'xtick', [])
set(gca, 'box', 'off', 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 22,'YAxisLocation','right')
ax1 = gca;
ax1.XColor = [0.5, 0.5, 0.5]; ax1.YColor = [0.5, 0.5, 0.5];
ax1_pos = ax1.Position;
ax2 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right', 'Color', 'none');
cmap = jet(sum(SL_weights~=0));
cIndex = 0;
for res = 1:size(SCFM_firstLayer_0, 1) % B-splines
    if SL_weights(res) > 0
        cIndex = cIndex + 1;
        plotTemp = squeeze(FMask_second_matix2(res, :, :));
        plot(plotTemp, 'color', cmap(cIndex, :), 'lineWidth', 2); hold on;
        xlim([0, 2000]); ylim([max(yRange_1-0.05, 0), yRange_2+0.05])
        set(gca, 'box', 'off')
        set(gca, 'Color', 'none')
    end
end
xticks([0, 250 ,500, 750, 1000, 1250, 1500, 1750, 2000])
xticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
% xlabel('Decoding Window (s)');
set(gca, 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 22)