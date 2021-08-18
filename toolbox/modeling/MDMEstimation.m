function [CM_outSample, c_predict_outSample, B_outSample, FitInfo_outSample, CrossValSet_0, CrossValSet_1, c_probability_outSample, theDeviance_outSample_fold] = MDMEstimation(inputFeature, target)
%% This function used to separate the data set into Training and Testing Sets which contain equal number of 1 and 0 labels
% And use this method to compare the fitting parameters lambda from overall
% data set with picked lambda from each fold
% Written by Xiwei 2017-9-18
warning off;
%% Firstly, do the classification of the whole dataset and get the overall lambda

lambdaPool = power(10, 1:-0.25:-5); % Define the lambda pool

%% Secondly, do the classification of each fold to make pure out of sample classification

zeroIndex = find(target==0); % Find those 0 labels
oneIndex = find(target==1); % Find those 1 labels

% Decide which group is the minority set
loopNum = min(length(zeroIndex), length(oneIndex));

% Label 0 group
zeroSetC = target(zeroIndex, :);
zeroSetP = inputFeature(zeroIndex, :);

% Label 1 group
oneSetC = target(oneIndex, :);
oneSetP = inputFeature(oneIndex, :);

if ((length(zeroIndex)==1) || (length(oneIndex)==1) || (isempty(zeroIndex)) || (isempty(oneIndex)))
    CM_outSample = 0; c_predict_outSample = zeros(length(target),1); goodFold = 0; B_outSample = []; FitInfo_outSample = [];
    CrossValSet_0 = []; CrossValSet_1 = []; c_probability_outSample = zeros(length(target),1); theDeviance_outSample_fold = [];
    
    if (size(CM_outSample,1)==1 && size(CM_outSample,2)==1)
        CM_outSample = [CM_outSample(1,1) 0;0 0];
    end
    return;
elseif (((1 < length(zeroIndex)) && (length(zeroIndex) < 10)) || ((1 < length(oneIndex)) && (length(oneIndex) < 10)))
    CrossValSet_0 = cvpartition(length(zeroSetC),'KFold', loopNum);
    CrossValSet_1 = cvpartition(length(oneSetC),'KFold', loopNum);
else
    CrossValSet_0 = cvpartition(length(zeroSetC),'KFold', 10);
    CrossValSet_1 = cvpartition(length(oneSetC),'KFold', 10);
end

% Predict label
c_predict_outSample = zeros(length(target), 1);
c_probability_outSample = zeros(length(target), 1);

% Count How many fold give good result
goodFold = 0;

if loopNum > 10
    loopNum = 10;
end

% Store all B and FitInfo from each fold
B_outSample = cell(loopNum, 1);
FitInfo_outSample = cell(loopNum, 1);

for loopFold = 1:loopNum
    % Label 0 Data Separation
    P0_train = zeroSetP(training(CrossValSet_0, loopFold),:);
    c0_train = zeroSetC(training(CrossValSet_0, loopFold),:);
    P0_test = zeroSetP(test(CrossValSet_0, loopFold),:);
    c0_test = zeroSetC(test(CrossValSet_0, loopFold),:);
    index0 = test(CrossValSet_0, loopFold)==1;
    % Below part is for check whether the code is right
    %                 P0_test = zeroSetP(training(CrossValSet_0, loopFold),:);
    %                 c0_test = zeroSetC(training(CrossValSet_0, loopFold),:);
    %                 index0 = training(CrossValSet_0, loopFold)==1;
    
    % Label 1 Data Separation
    P1_train = oneSetP(training(CrossValSet_1, loopFold),:);
    c1_train = oneSetC(training(CrossValSet_1, loopFold),:);
    P1_test = oneSetP(test(CrossValSet_1, loopFold),:);
    c1_test = oneSetC(test(CrossValSet_1, loopFold),:);
    index1 = test(CrossValSet_1, loopFold)==1;
    % Below part is for check whether the code is right
    %                 P1_test = oneSetP(training(CrossValSet_1, loopFold),:);
    %                 c1_test = oneSetC(training(CrossValSet_1, loopFold),:);
    %                 index1 = training(CrossValSet_1, loopFold)==1;
    
    % Find the original index in c for those testing label
    originalCIndex = [zeroIndex(index0); oneIndex(index1)];
    
    % Training Set
    P_train = [P0_train;P1_train];
    c_train = [c0_train;c1_train];
    %     shuffleIndex_train = randperm(length(c_train))';% Shuffle
    shuffleIndex_train = 1:length(c_train);% Without Shuffle
    P_train = P_train(shuffleIndex_train,:);
    c_train = c_train(shuffleIndex_train,:);
    
    % Testing Set
    P_test = [P0_test;P1_test];
    c_test = [c0_test;c1_test];
    %     shuffleIndex_test = randperm(length(c_test))';% Shuffle
    shuffleIndex_test = 1:length(c_test);% Without Shuffle
    P_test = P_test(shuffleIndex_test,:);
    c_test = c_test(shuffleIndex_test,:);
    originalCIndex = originalCIndex(shuffleIndex_test);
    
    % Train Lassoglm
    [B_outSample_fold,FitInfo_outSample_fold] = lassoglm(P_train,double(c_train),'binomial', 'Lambda', lambdaPool, 'MaxIter', 1e3);
    B_outSample{loopFold} = sparse(B_outSample_fold);  % Store fitting info - B
    FitInfo_outSample{loopFold} = FitInfo_outSample_fold; % Store fitting info - FitInfo
    C_v = B_outSample_fold; % selected coefficients
    C0 = FitInfo_outSample_fold.Intercept;
    theDeviance_outSample_fold = zeros(size(C_v, 2), 1);
    c_p_outSample_fold = zeros(size(C_v, 2), size(c_test, 1));
    
    for loopLambda = 1:size(C_v, 2) % Loop through all lambda
        c_i_outSample = P_test * C_v(:, loopLambda) + C0(loopLambda);
        c_p_outSample_fold(loopLambda, :) = 1 ./ (1 + exp(-c_i_outSample));
        probability_outSample_temp = (1 - c_test) + 2 * (c_test - 0.5) .* c_p_outSample_fold(loopLambda, :)';
        theDeviance_outSample_fold(loopLambda) = -2 * sum(log(probability_outSample_temp));
    end
   
    % Then find the minimum deviance and it's parameters
    minDeviance = min(theDeviance_outSample_fold);
    indexDev = find(theDeviance_outSample_fold == minDeviance);
    if length(indexDev) > 1
        indexDev = indexDev(1);
    end
    sc_p = c_p_outSample_fold(indexDev, :);
    c_predict_fold = double(sc_p > 0.5);
    CM_outSample = confusionmat(c_test, c_predict_fold);

    if (size(CM_outSample,1)==1&&size(CM_outSample,2)==1)
        CM_outSample = [CM_outSample(1,1) 0;0 0];
    end
    MCC_outSample = mcc(CM_outSample); % conffusion matrix
    
    if(MCC_outSample > 0.3)
        goodFold = goodFold + 1;
    end
    
    c_predict_outSample(originalCIndex) = c_predict_fold;
    c_probability_outSample(originalCIndex) = sc_p;
    
end
CM_outSample = confusionmat(double(target),c_predict_outSample); % confusion matrix

if (size(CM_outSample,1)==1 && size(CM_outSample,2)==1)
    CM_outSample = [CM_outSample(1,1) 0;0 0];
end