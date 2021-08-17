%% This script is used for checking the SCFM of single-neuron simulation cases
% Corresponding to section 3.1 (Fig.6) in the manuscript of NECO-2021-She
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-11-26
clear; clc;
addpath(genpath('../toolbox'));

%% Comment/uncomment the corresponding case you want to check
% Case 1 - single neuron & low std & low intensity
nTrials = 100;
runCase = 'SN&LS4';
nCategories = 2;
globalIndex = 4;
Subject = 'SingleNeuron';
plotTitle = 'B-Spline Fitting Low-resolution Neuron';
plotTitle2 = 'Simulated Neuron with Low Temporal Resolution';

% % Case 2 - single neuron & high std & low intensity
% nTrials = 100;
% runCase = 'SN&HS4';
% nCategories = 2;
% globalIndex = 3;
% Subject = 'SingleNeuron';
% plotTitle = 'B-Spline Fitting High-resolution Neuron';
% plotTitle2 = 'Simulated Neuron with High Temporal Resolution';

Category = 'A';


%% load data and results
iF1 = strcat('Results\',Subject, '&', mat2str(nCategories),'Categories\NestedCVDLMDM_synthetic_',Subject, '&', mat2str(nCategories),'Categories_', Category,'_', runCase,'_globalSelected_rep.mat');
load(iF1);

iF3 = strcat('Synthetic_Input\SimulatedRawData\DLMDM_SyntheticInputData_',Subject, '_', mat2str(nCategories),'Categories_', runCase,'.mat');
load(iF3, 'spikeTensor', 'neuronSpikes', 'neuronPIFs');
SpikeTensor = spikeTensor;

%% Get variables
numNeuron = 1;
resolutionPool = MDfit.m_all;
L = MDfit.L; % Length of decoding window 2000
order = MDfit.d; % Order 3
numFold1 = length(MDfit.R_first(1, 1).FL_inside_Coefficients);

FMask_first_matrix_folds = zeros(L+1, numNeuron, numFold1);
BSpline_first_fold = zeros(L+1, 154, numFold1);
BSpline_first_fold2 = zeros(L+1, 154, numFold1);
probFMask_first_matrix_folds = zeros(L+1, numNeuron, numFold1);
probFMask_first_matrix = zeros(length(resolutionPool), L+1, numNeuron);
probC0_first_matrix = zeros(length(resolutionPool), 1, numNeuron);
Bspline_first = zeros(length(resolutionPool), L+1, 154);
Bspline_first2 = zeros(length(resolutionPool), L+1, 154);
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
        weights2_firstLayer(weights2_firstLayer<0) = 0; % Here we only consider positive patterns
        % Get the Functional Matrics - Eq. 15
        F_maskTemp = BSpline * weights2_firstLayer; % F is always with the weights
        
        % Fold averaged the F
        FMask_first_matrix_folds(:, :, fold) = F_maskTemp;
        
        % For bspline visualization
        BSpline_first_fold(:, 1:size(BSpline, 2), fold) = ( 1./(1+ exp(-BSpline .* weights2_firstLayer(:, 1)' - w0_firstLayer)) ) ;
        BSpline_first_fold(:, size(BSpline, 2)+1:end, fold) = 1 ./ (1+ exp(-w0_firstLayer) );
        Bspline_first_w0_folds(fold) = 1 ./ (1 + exp(-w0_firstLayer));
        
    end
    % Calculate the probability matrices - Multi Trial Multi Resolution
    probTemp = mean(FMask_first_matrix_folds, 3); % Un-weighted B-spline
    probTemp2 = mean(BSpline_first_fold, 3); % Un-weighted B-spline
    Bspline_first(resolutionIndex, :, :) = probTemp2;
    Bspline_first_w0(resolutionIndex, 1) = mean(Bspline_first_w0_folds);
    
    w0Temp = mean(w0_firstLayerPool);
    tempSCFM_first = 1 ./ (1+ exp(-w0Temp - probTemp) );
    probFMask_first_matrix(resolutionIndex, :, :) = tempSCFM_first;
    probC0_first_matrix(resolutionIndex, :, :) = 1 ./ (1+ exp(-w0Temp) );
end

% Ensemble original first-layer SCFM
SCFM_firstLayer_0 = probFMask_first_matrix;
SCFM_firstLayer = mean(SCFM_firstLayer_0, 1);

% Visualize first layer averaged SCFM
% figure('Position',[50 50 1400 700]);
% hSCFM = area(SCFM_firstLayer);
% hSCFM.FaceColor = [0.68, 0.68, 0.68];
% hSCFM.LineStyle = 'none';
% ylabel('SCFM')

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
C0_second_prob_matrix = zeros(numFold, L+1, numNeuron);
SL_C0 = mean(SL_outside_GlobalC0, 1);

for f = 1:numFold
    w0Temp = SL_outside_GlobalC0(f);
    weightsTemp = SL_outside_GlobalCoefficients(f, :);
    
    % Consider intecept from all other classifier
    inteceptTemp_SCFM = weightsTemp * probC0_first_matrix;
   
    FMask_second_0 = SL_C0 + probFMask_first_permute' * weightsTemp';
    FMask_second = reshape(FMask_second_0, L+1, []); % DecodingWindow * Neuron
    
    FMask_second_prob_matrix(f, :, :) = 1 ./ (1 + exp(-FMask_second) ); % F to Prob for this Fold
end
% F to Prob for this trial by averaged the fold prob
probFMask_second_matrix = squeeze(mean(FMask_second_prob_matrix, 1));

%% Try visual
SL_weights = mean(SL_outside_GlobalCoefficients, 1);
SL_C0 = mean(SL_outside_GlobalC0, 1);
FMask_second_matix = zeros(length(resolutionPool), size(Bspline_first, 2), size(Bspline_first, 3));
FMask_second_matix2 = zeros(length(resolutionPool), size(Bspline_first, 2), size(Bspline_first, 3));
for res = 1:length(resolutionPool)
    FMask_first_matrix = squeeze(Bspline_first(res, :, :));
    
    % Consider intecept from all other classifier
    weights_temp = SL_weights; 
    weights_temp(res) = [];
    probC0_first_temp = Bspline_first_w0;
    probC0_first_temp(res) = [];
    inteceptTemp = weights_temp * probC0_first_temp;
    
    FMask_second_matix(res, :, :) = 1 ./ (1 + exp( -FMask_first_matrix .* SL_weights(res) - SL_C0 - inteceptTemp) );
end

% SCFM
SCFM_Prob_final = probFMask_second_matrix;
SCFM_Prob_final_vis = SCFM_Prob_final;
SCFM_Prob_final_vis(SCFM_Prob_final_vis<0) = 0; % In simulation, we only consider positive patterns


%% Start visual
% PIFs and Histogram - spatio-temporal patterns
dt = 1/500; % Step size, unit: ms
tSim = 4; % how many seconds we are going to simulate
fr_base = 5; % Baseline firing rate, unit: Hz
L = tSim/dt; % length of pattern
t =[1:1:L]'*dt; % time

SP1 = squeeze(neuronSpikes(1, :, :));
histBinSize = 20; % N * 2ms
SP1_binned = zeros(size(SP1, 1)/histBinSize, size(SP1, 2));
for b = 1:size(SP1_binned, 1)
    SP1_binned(b, :) = sum(SP1((b*histBinSize-histBinSize+1):(b*histBinSize), :));
end
hist1 = sum(SP1_binned, 2)  / nTrials / histBinSize;
t_hist = [1:size(SP1_binned)] * dt * histBinSize;

for c = 1
    figure('Position',[50 50 1400 700]);
    p = squeeze(squeeze(neuronPIFs(c, :)));
    line(t,p, 'Color', 'r', 'lineWidth', 5); hold on; % xlabel('Simulated Time (s)'); ylabel('Probability Intensity Function');
    set(gca,'xtick',[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], 'xticklabel', [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]); %ylim([0, 0.05])
    stairs(t_hist, hist1, 'LineWidth', 3)
    set(gca, 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 26)
    set(gca,'xtick',[]);
    ax1 = gca;
    ax1.XColor = 'r'; ax1.YColor = 'r';
    ax1_pos = ax1.Position;
    ax2 = axes('Position',ax1_pos,'XAxisLocation','bottom','YAxisLocation','right', 'Color', 'none');
    x = squeeze(squeeze(neuronSpikes(c, :, :)));
    rasterplot(x'); % ylabel('Simulated Trials');
    % title(plotTitle2)
    ax2.XColor = 'k'; ax2.YColor = 'k';
    % set(gca,'xtick',[]);
    set(gca, 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 26)
end

% SCFM & Bsplines
yRange_1 = min( min(min(SCFM_Prob_final_vis)), min(min(min(FMask_second_matix))) );
yRange_2 = max( max(max(SCFM_Prob_final_vis)), max(max(max(FMask_second_matix))) );
offsetSCFM = min(min(SCFM_Prob_final_vis)) - min(min(min(FMask_second_matix)));
figure('Position',[50 50 1400 700]);
hSCFM = area(SCFM_Prob_final_vis(:, :)-offsetSCFM);
hSCFM.FaceColor = [0.68, 0.68, 0.68];
hSCFM.LineStyle = 'none';
% ylabel('SCFM')
xlim([0, 2000])
ylim([yRange_1-0.05, yRange_2+0.05])
set(gca, 'xtick', [])
set(gca, 'box', 'off', 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 26,'YAxisLocation','right')
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
        plot(plotTemp, 'color', cmap(cIndex, :), 'lineWidth', 5); hold on;
        xlim([0, 2000]); ylim([yRange_1-0.05, yRange_2+0.05])
        set(gca, 'box', 'off')
        set(gca, 'Color', 'none')
        if cIndex == 1
%             ylabel('Bspline Probability Functional Curves')
        end
    end
end
xticks([0, 250 ,500, 750, 1000, 1250, 1500, 1750, 2000])
xticklabels([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
% xlabel('Decoding Window (s)');
set(gca, 'FontName', 'Arial','FontWeight','bold', 'Fontsize', 26)