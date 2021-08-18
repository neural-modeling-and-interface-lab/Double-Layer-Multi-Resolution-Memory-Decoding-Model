classdef DLMDM_Simulated_combinedNeuron_minRes
    %This runs memory decoding for a given category, patient, and fitting
    %options
    
    properties
        Subject % patient folder
        Category % category for regression
        oF % output file
        
        % fitting options
        Num_Trial %number of random trials
        m_all %array with number of b splines
        d % number of b spline knots
        L %memory length on either side of the event in ms
        par %this indicates if fitting is done in a parfor loop or not
        
        % intermediate variables
        target %this is the category classification used in estimation
        c_ctrl %this is the category classification used in all control trials
        R_exp %this stores the results of the non-control data
        R_ctrl %this stors the results of the control categories
        tFit %elapsed fitting time
        R_first % FirstLayer MD Results
        R_second % SecondLayer MD Results
        
        % Nest CV variables
        TrainingSet_SpikeTensor
        TrainingSet_target
        TestingSet_SpikeTensor
        TestingSet_target
        CrossValSet
    end
    
    methods
        function obj = DLMDM_Simulated_combinedNeuron_minRes(Subject, Category, currentParition, nCategories, runCase, varargin)
            [obj.Num_Trial, obj.m_all, obj.d,obj.L, obj.par] = process_options(varargin,...
                'Num_Trial',32,'m_all',10:200,'d',3,'L',2000,'par',0);
            warning off;
            %% Load Data
            iF1 = strcat('Synthetic_Input\CombinedNeuron&', mat2str(nCategories), 'Categories','\NestCVDLMDM_synthetic_CombinedNeuron&', mat2str(nCategories),'Categories_',Category,'_fold_', mat2str(currentParition),'_', runCase,'.mat');
            load(iF1);
            
            %store object properties
            obj.target = target;
            obj.Category = Category;
            obj.Subject = Subject;
            obj.TrainingSet_SpikeTensor = TrainingSet_SpikeTensor;
            obj.TrainingSet_target = TrainingSet_target;
            obj.TestingSet_SpikeTensor = TestingSet_SpikeTensor;
            obj.TestingSet_target = TestingSet_target;
            obj.CrossValSet = CrossValSet;
            
            %specify output file and save memory decoding setup
            obj.oF = strcat('Results\CombinedNeuron&', mat2str(nCategories), 'Categories','\Raw Results\NestedCVDLMDM_synthetic_CombinedNeuron&', mat2str(nCategories),'Categories_',Category,'_part',mat2str(currentParition), '_', runCase,'_minRes.mat');
            MDfit = obj;
            save(obj.oF,'MDfit')
        end
        function SpikeTensor = getSpikeTensor(obj)
            SpikeTensor = obj.TrainingSet_SpikeTensor;
        end
        function [firstR, secondR] = runTrial(obj,ti,IfNotCtrl,SpikeTensor,varargin)
            % Training output labels
            thisTrial_c = obj.TrainingSet_target;
            
            % Testing input and output
            testingTensor = obj.TestingSet_SpikeTensor;
            testingTarget = obj.TestingSet_target;
            
            %get all parameters from object used in the fit of each trial
            Num_Trial = obj.Num_Trial;
            m_all = obj.m_all;
            d = obj.d;
            observes_secondLayer = zeros(length(thisTrial_c), length(m_all));
            observes_secondLayer_testing = zeros(length(testingTarget), length(m_all));
            %% First Layer MD training
            for mi = 1:length(m_all)
                m = m_all(mi);
                P = SpikeTensor2BSplineFeatureMatrix(SpikeTensor, m, d);
                
                % Memory Decoding Model Estimation
                [FL_inside_ConfusionMatrix, FL_inside_predictions, FL_inside_Coefficients, FL_inside_FitInfo, FL_inside_CrossValSet_0, FL_inside_CrossValSet_1, FL_inside_probabilities, FL_inside_Deviance] = MDMEstimation(P, thisTrial_c);
                
                % Save first layer inside results
                firstR(mi).Resolution = mi; % Resolution Number
                firstR(mi).FL_inside_ConfusionMatrix = FL_inside_ConfusionMatrix; % confusion matrix
                firstR(mi).FL_inside_MCC = mcc(FL_inside_ConfusionMatrix); % MCC
                firstR(mi).FL_inside_predictions = FL_inside_predictions; % prediction - Out Sample
                firstR(mi).FL_inside_Coefficients = FL_inside_Coefficients; % Fiting Results for each fold - Coefficients
                firstR(mi).FL_inside_FitInfo = FL_inside_FitInfo; % Fiting Results for each fold - Fitting Information
                firstR(mi).FL_inside_CrossValSet_0 = FL_inside_CrossValSet_0; % Patition setting for label 0
                firstR(mi).FL_inside_CrossValSet_1 = FL_inside_CrossValSet_1; % Patition setting for label 1
                firstR(mi).FL_inside_probabilities = FL_inside_probabilities; % prediction - probability
                firstR(mi).FL_inside_Deviance = FL_inside_Deviance;
                firstR(mi).FL_inside_Target = thisTrial_c;
                
                % Pick the global lambda
                globalMinDeviance = min(FL_inside_Deviance);
                globalIndex = find(FL_inside_Deviance == globalMinDeviance);
                if length(globalIndex) > 1
                    globalIndex = globalIndex(1);
                end
                
                %% First Layer Nested testing
                P_test = SpikeTensor2BSplineFeatureMatrix(testingTensor, m, d);
                
                numFold = length(FL_inside_Coefficients);
                FL_outside_probabilities = zeros(numFold, length(testingTarget));
                FL_outside_predictions = zeros(numFold, length(testingTarget));
                FL_outside_MCC = zeros(numFold, 1);
                FL_outside_GlobalCoefficients = zeros(numFold, size(FL_inside_Coefficients{1}, 1));
                FL_outside_GlobalC0 = zeros(numFold, 1);
                for tempI = 1:numFold
                    
                    globalC0 = FL_inside_FitInfo{tempI}.Intercept(globalIndex);
                    FL_outside_GlobalC0(tempI) = globalC0;
                    FL_outside_GlobalCoefficients(tempI, :) = FL_inside_Coefficients{tempI}(:, globalIndex);
                    c_i = P_test * FL_inside_Coefficients{tempI}(:, globalIndex) + globalC0;
                    c_p = 1 ./ (1 + exp(-c_i));
                    FL_outside_probabilities(tempI, :) = c_p;
                    FL_outside_predictions(tempI, :) = double(c_p>=0.5); % Can adjust this threshold latter
                    FL_outside_CM = confusionmat(testingTarget, double(c_p>=0.5));
                    if (size(FL_outside_CM,1)==1&&size(FL_outside_CM,2)==1)
                        FL_outside_CM = [FL_outside_CM(1,1) 0;0 0];
                    end
                    FL_outside_MCC(tempI, 1) = mcc(FL_outside_CM);
                end
                
                % Average the predicted probabilities of all folds
                FL_outside_probabilities_foldAveraged = mean(FL_outside_probabilities, 1);
                FL_outside_CM_foldAveraged = confusionmat(testingTarget, double(FL_outside_probabilities_foldAveraged>=0.5));
                if (size(FL_outside_CM_foldAveraged,1)==1&&size(FL_outside_CM_foldAveraged,2)==1)
                    FL_outside_CM_foldAveraged = [FL_outside_CM_foldAveraged(1,1) 0;0 0];
                end
                FL_outside_MCC_foldAveraged = mcc(FL_outside_CM_foldAveraged);
                
                % Save FL testing results
                firstR(mi).FL_outside_GlobalCoefficients = FL_outside_GlobalCoefficients;
                firstR(mi).FL_outside_GlobalC0 = FL_outside_GlobalC0;
                firstR(mi).FL_outside_probabilities = FL_outside_probabilities;
                firstR(mi).FL_outside_predictions = FL_outside_predictions;
                firstR(mi).FL_outside_MCCs = FL_outside_MCC;
                firstR(mi).FL_outside_probabilities_foldAveraged = FL_outside_probabilities_foldAveraged;
                firstR(mi).FL_outside_CM_foldAveraged = FL_outside_CM_foldAveraged;
                firstR(mi).FL_outside_MCC_foldAveraged = FL_outside_MCC_foldAveraged;
                firstR(mi).FL_outside_Target = testingTarget;
                
                % Show Results - First Layer
                %                 pause(rand*0.01);
                fprintf('T:%d; Kts:%d; FL Inside CM:%d %d %d %d; FL Inside MCC:%1.2f\n FL Outside FoldAveraged CM:%d %d %d %d; FL Outside FoldAveraged MCC:%1.2f\n',ti, m, FL_inside_ConfusionMatrix, mcc(FL_inside_ConfusionMatrix), FL_outside_CM_foldAveraged, FL_outside_MCC_foldAveraged);
                
                % Save the probability for the second MD
                observes_secondLayer(:, mi) = FL_inside_probabilities;
                observes_secondLayer_testing(:, mi) = FL_outside_probabilities_foldAveraged; % Now take the averaged probability
            end
            
            %% Second Layer MD training
            % Memory Decoding Model Estimation
            [SL_inside_ConfusionMatrix, SL_inside_predictions, SL_inside_Coefficients, SL_inside_FitInfo, SL_inside_CrossValSet_0, SL_inside_CrossValSet_1, SL_inside_probabilities, SL_inside_Deviance] = MDMEstimation(observes_secondLayer, double(thisTrial_c));
            % Save second layer inside results
            secondR.SL_inside_ConfusionMatrix = SL_inside_ConfusionMatrix; % confusion matrix
            secondR.SL_inside_MCC = mcc(SL_inside_ConfusionMatrix); % MCC
            secondR.SL_inside_predictions = SL_inside_predictions; % prediction - Out Sample
            secondR.SL_inside_Coefficients = SL_inside_Coefficients; % Fiting Results for each fold - Coefficients
            secondR.SL_inside_FitInfo = SL_inside_FitInfo; % Fiting Results for each fold - Fitting Information
            secondR.SL_inside_CrossValSet_0 = SL_inside_CrossValSet_0; % Patition setting for label 0
            secondR.SL_inside_CrossValSet_1 = SL_inside_CrossValSet_1; % Patition setting for label 1
            secondR.SL_inside_probabilities = SL_inside_probabilities; % prediction - probability
            secondR.SL_inside_Deviance = SL_inside_Deviance;
            
            %% Second Layer MD Nested testing
            % Test with the self CV method
            % treat every model as different trals
            % Pick the global lambda
            globalMinDeviance2 = min(SL_inside_Deviance);
            globalIndex2 = find(SL_inside_Deviance == globalMinDeviance2);
            if length(globalIndex2) > 1
                globalIndex2 = globalIndex2(1);
            end
                
            numFold = length(SL_inside_Coefficients);
            SL_outside_probabilities = zeros(numFold, length(testingTarget));
            SL_outside_predictions = zeros(numFold, length(testingTarget));
            SL_outside_MCC = zeros(numFold, 1);
            SL_outside_GlobalCoefficients = zeros(numFold, size(SL_inside_Coefficients{1}, 1));
            SL_outside_GlobalC0 = zeros(numFold, 1);
            for tempI = 1:numFold
                
                globalC0 = SL_inside_FitInfo{tempI}.Intercept(globalIndex2);
                SL_outside_GlobalC0(tempI) = globalC0;
                SL_outside_GlobalCoefficients(tempI, :) = SL_inside_Coefficients{tempI}(:, globalIndex2);
                c_i2 = observes_secondLayer_testing * SL_inside_Coefficients{tempI}(:, globalIndex2) + globalC0;
                c_p2 = 1 ./ (1 + exp(-c_i2));
                SL_outside_probabilities(tempI, :) = c_p2;
                SL_outside_predictions(tempI, :) = double(c_p2>=0.5); % Can adjust this threshold latter
            end
            
            % treat every model(fold) as different trals
            SL_outside_probabilities_modelEnlarged = SL_outside_probabilities(:);
            SL_outside_predictions_modelEnlarged = double(SL_outside_probabilities_modelEnlarged>=0.5);
            testingTarget_modelEnlarged = testingTarget;
            for enlarge = 1:size(SL_outside_probabilities, 1)-1
                testingTarget_modelEnlarged = [testingTarget_modelEnlarged; testingTarget];
            end
            SL_outside_CM_modelEnlarged = confusionmat(testingTarget_modelEnlarged, SL_outside_predictions_modelEnlarged);
            if (size(SL_outside_CM_modelEnlarged,1)==1&&size(SL_outside_CM_modelEnlarged,2)==1)
                SL_outside_CM_modelEnlarged = [SL_outside_CM_modelEnlarged(1,1) 0;0 0];
            end
            SL_outside_MCC_modelEnlarged = mcc(SL_outside_CM_modelEnlarged);
            
            % Or take the mean of models
            SL_outside_probabilities_modelAveraged = mean(SL_outside_probabilities, 1);
            SL_outside_predictions_modelAveraged = double(SL_outside_probabilities_modelAveraged>=0.5);
            SL_outside_CM_modelAveraged = confusionmat(testingTarget, SL_outside_predictions_modelAveraged);
            if (size(SL_outside_CM_modelAveraged,1)==1&&size(SL_outside_CM_modelAveraged,2)==1)
                SL_outside_CM_modelAveraged = [SL_outside_CM_modelAveraged(1,1) 0;0 0];
            end
            SL_outside_MCC_modelAveraged = mcc(SL_outside_CM_modelAveraged);
            
            % Save secondMD outside results
            secondR.SL_outside_probabilities = SL_outside_probabilities;
            secondR.SL_outside_predictionss = double(SL_outside_probabilities>=0.5);
            secondR.SL_outside_MCC_modelEnlarged = SL_outside_MCC_modelEnlarged; % MCC 1
            secondR.SL_outside_predictions_modelEnlarged = double(SL_outside_predictions_modelEnlarged); % prediction 1
            secondR.SL_outside_probabilities_modelEnlarged = SL_outside_probabilities_modelEnlarged; % probability 1
            secondR.testingTarget_modelEnlarged = testingTarget_modelEnlarged; % target 1
            secondR.SL_outside_MCC_modelAveraged = SL_outside_MCC_modelAveraged; % MCC 2 
            secondR.SL_outside_predictions_modelAveraged = double(SL_outside_predictions_modelAveraged); % prediction 2
            secondR.SL_outside_probabilities_modelAveraged = SL_outside_probabilities_modelAveraged; % probability 2
            secondR.testingTarget = testingTarget; % target 2
            
            % Show Results - Second Layer
            pause(rand*0.01);
            fprintf('T:%d; SL Inside CM:%d %d %d %d; SL Inside MCC:%1.2f\n SL Outside TrialEnlarged CM:%d %d %d %d; SL Outside TrialEnlarged MCC:%1.2f\n '...
                ,ti, SL_inside_ConfusionMatrix, mcc(SL_inside_ConfusionMatrix), SL_outside_CM_modelEnlarged, SL_outside_MCC_modelEnlarged, SL_outside_CM_modelAveraged, SL_outside_MCC_modelAveraged);
        end
        function [firstR, secondR] = runAllTrials(obj,IfNotCtrl,varargin)
            IfPlot = process_options(varargin,'IfPlot',0);
            SpikeTensor=obj.getSpikeTensor;
            if obj.par
                if IfPlot
                    warning('Plotting not supported for parallel')
                end
                parfor ti = 1:obj.Num_Trial
                    [R_sepTrial{ti}, R2_sepTrial{ti}] = obj.runTrial(ti,IfNotCtrl,SpikeTensor);
                end
            else
                for ti = 1:obj.Num_Trial
                    [R_sepTrial{ti}, R2_sepTrial{ti}] = obj.runTrial(ti,IfNotCtrl,SpikeTensor,'IfPlot',IfPlot);
                end
            end
            %Put R back in the original format
            for ti = 1:obj.Num_Trial
                % prepare spatio-temporal patterns for classification
                for mi = 1:length(obj.m_all)
                    firstR(ti,mi) = R_sepTrial{ti}(mi);
                end
                secondR(ti) = R2_sepTrial{ti};
            end
            
        end
        function MDfit = run(obj,varargin)
            if obj.par
                poolOb = obj.setupPool;
            end
            tStart = tic;
            [IfPlot] = process_options(varargin,'IfPlot',0);
            IfNotCtrl = 1;
            [obj.R_first, obj.R_second] = obj.runAllTrials(IfNotCtrl,'IfPlot',IfPlot); % Experiment Group
            
            tFit = toc(tStart);
            obj.tFit = tFit;
            MDfit = obj;
            save(obj.oF,'MDfit', '-v7.3')
            if obj.par
                poolOb.delete;
            end
        end
        function [MCC_all_exp,MCC_all_ctrl] = getMCC(obj)
            %% Gets MCC from control and experimental categories
            [NumTrial, NumRes] = size(obj.R_exp);
            MCC_all_exp = zeros(NumTrial,NumRes);
            for i = 1:NumTrial
                for j = 1:NumRes
                    MCC_all_exp(i,j)=obj.R_exp(i,j).MCC;
                end
            end
            MCC_all_ctrl = zeros(NumTrial,NumRes);
            for i = 1:NumTrial
                for j = 1:NumRes
                    MCC_all_ctrl(i,j)=obj.R_ctrl(i,j).MCC;
                end
            end
        end
        function poolOb = setupPool(obj)
            if ~isunix
                poolOb = parpool;
            else  %here, the default is to put a seperate node per trial, with the number of workers equal to the number of trials
                nWorkers = obj.Num_Trial;
                % ------------------------ New for Slurm ------------------
                clusProf = get_SLURM_cluster('/home/rcf-proj/tb/xiweishe/matlab_storage','/usr/usc/matlab/R2018a/SlurmIntegrationScripts','--time=50:00:00 --partition berger');
                % ------------------------ New for Slurm ------------------
                poolOb = parpool(clusProf,nWorkers);
            end
        end
    end
    
end

