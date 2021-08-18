classdef MemoryDecode_algorithmComparison_simulation
    %This runs memory decoding for a given category, patient, and fitting
    %options
    
    properties
        DataFolder % patient folder
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
        function obj = MemoryDecode_algorithmComparison_simulation(Category, currentParition, varargin)
            [obj.Num_Trial, obj.m_all, obj.d,obj.L, obj.par] = process_options(varargin,...
                'Num_Trial',32,'m_all',10:200,'d',3,'L',2000,'par',0);
            warning off;
            %% Load Data
            if sum(Category == '1C') == 2 % 30N&500T&5C
                iF1 = strcat('Synthetic_Input\AlgorithmComparison\30N&500T&5C\NestCVDLMDM_synthetic_Realistic_30N&500T&5C_CategoryA', mat2str(currentParition),'.mat');
            elseif sum(Category == '2C') == 2 % 20N&200T&2C
                iF1 = strcat('Synthetic_Input\AlgorithmComparison\20N&200T&2C\NestCVDLMDM_synthetic_Realistic_20N&200T&2C_', mat2str(currentParition),'.mat');
            else % 20N&200T&5C
                iF1 = strcat('Synthetic_Input\AlgorithmComparison\20N&200T&5C\NestCVDLMDM_synthetic_Realistic_20N&200T&', Category, '_', mat2str(currentParition),'.mat');
            end
            load(iF1);
            
            %store object properties
            obj.target = target;
            obj.TrainingSet_SpikeTensor = TrainingSet_SpikeTensor;
            obj.TrainingSet_target = TrainingSet_target;
            obj.TestingSet_SpikeTensor = TestingSet_SpikeTensor;
            obj.TestingSet_target = TestingSet_target;
            
            %specify output file and save memory decoding setup
            if sum(Category == '1C') == 2 % 30N&500T&5C
                obj.oF = strcat('Results\AlgorithmComparison\30N&500T&5C\AlgorithmComparison_30N&500T&5C_CategoryA_part',mat2str(currentParition), '_rep.mat');
            elseif sum(Category == '2C') == 2 % 20N&200T&2C
                obj.oF = strcat('Results\AlgorithmComparison\20N&200T&2C\AlgorithmComparison_20N&200T&2C_part',mat2str(currentParition), '_rep.mat');
            else % 20N&200T&5C
                obj.oF = strcat('Results\AlgorithmComparison\20N&200T&2C\AlgorithmComparison_20N&200T&', Category,'_part',mat2str(currentParition), '_rep.mat');
            end
            MDfit = obj;
            save(obj.oF,'MDfit')
        end
        function SpikeTensor = getSpikeTensor(obj)
            SpikeTensor = obj.TrainingSet_SpikeTensor;
        end
        function thisR = runTrial(obj,ti,~,SpikeTensor,varargin)
            warning off;
            % Training output labels
            thisTrial_c = obj.TrainingSet_target;
            
            % Testing input and output
            testingTensor = obj.TestingSet_SpikeTensor;
            testingTarget = obj.TestingSet_target;
            
            % Lambda for L1-regularization
            lambdaPool = power(10, 1:-0.25:-5);
            
            %get all parameters from object used in the fit of each trial
            Num_Trial = obj.Num_Trial;
            m_all = obj.m_all;
            d = obj.d;
            
            % Original spike trains
            P_0 = SpikeTensor(:,:);
            P_0_test = testingTensor(:,:);
            
            % PCA features of spikes
            [P_coeff,P_score,P_latent] = pca(P_0);
            [P_coeff_test,P_score_test,P_latent_test] = pca(testingTensor(:,:));
            P_score = P_score(:, 1:size(P_score_test, 2));
            
            P_longVec = [];
            P_longVec_test = [];
            bestMCC_0 = -inf; bestMCC_1 = -inf; bestMCC_2 = -inf;
            for mi = 1:length(m_all)
                m = m_all(mi);
                P = SpikeTensor2BSplineFeatureMatrix(SpikeTensor, m, d);
                P_test = SpikeTensor2BSplineFeatureMatrix(testingTensor, m, d);
                
                % Cascades all resolution features into a long vector
                P_longVec = [P_longVec, P];
                P_longVec_test = [P_longVec_test, P_test];
                
                % Start algorithms
                
                % Algorithm #0 - Lasso without bagging
                tic
                tStar = tic;
                [B_0,FitInfo_0] = lassoglm(P, double(thisTrial_c),'binomial', 'Lambda', lambdaPool, 'CV', 10);
                idxLambdaMinDeviance_0 = FitInfo_0.IndexMinDeviance;
                B0_0 = FitInfo_0.Intercept(idxLambdaMinDeviance_0);
                coef_0 = [B0_0; B_0(:,idxLambdaMinDeviance_0)];
                prob_0 = glmval(coef_0, P_test, 'logit');
                predict_0 = double(prob_0>=0.5);
                CM_0 = confusionmat(testingTarget, predict_0);
                MCC_0 = mcc(CM_0);
                tElapsed_0 = toc(tStar);
                thisR(mi).prob_0 = prob_0; thisR(mi).CM_0 = CM_0; thisR(mi).MCC_0 = MCC_0; thisR(mi).tElapsed_0 = tElapsed_0;
                if MCC_0 > bestMCC_0
                    bestMCC_0 = MCC_0;
                end
                
                % Algorithm #1 - Logistic regression
                tic
                tStar = tic;
                [B_1,FitInfo_1] = lassoglm(P, double(thisTrial_c),'binomial', 'Lambda', 0, 'CV', 10);
                idxLambdaMinDeviance_1 = FitInfo_1.IndexMinDeviance;
                B0_1 = FitInfo_1.Intercept(idxLambdaMinDeviance_1);
                coef_1 = [B0_1; B_1(:,idxLambdaMinDeviance_1)];
                prob_1 = glmval(coef_1, P_test, 'logit');
                predict_1 = double(prob_1>=0.5);
                CM_1 = confusionmat(testingTarget, predict_1);
                MCC_1 = mcc(CM_1);
                tElapsed_1 = toc(tStar);
                thisR(mi).prob_1 = prob_1; thisR(mi).CM_1 = CM_1; thisR(mi).MCC_1 = MCC_1; thisR(mi).tElapsed_1 = tElapsed_1;
                if MCC_1 > bestMCC_1
                    bestMCC_1 = MCC_1;
                end
                
                % Algorithm #2 - Naive bayes
                tic
                tStar = tic;
                P_nb = P; P_nb(:,all(P==0,1))=[]; % Remove all zero feature
                P_test_nb = P_test; P_test_nb(:,all(P==0,1))=[];
                Md_nb = fitcnb(P_nb, double(thisTrial_c), 'DistributionNames', 'kernel');
                [predict_2,prob_2,~] = predict(Md_nb, P_test_nb);
                CM_2 = confusionmat(testingTarget, predict_2);
                MCC_2 = mcc(CM_2);
                tElapsed_2 = toc(tStar);
                thisR(mi).prob_2 = prob_2; thisR(mi).CM_2 = CM_2; thisR(mi).MCC_2 = MCC_2; thisR(mi).tElapsed_2 = tElapsed_2;
                if MCC_2 > bestMCC_2
                    bestMCC_2 = MCC_2;
                end
                
                fprintf('T:%d; Kts:%d; MCC lasso:%1.2f; MCC LR:%1.2f; MCC NB:%1.2f\n',ti, m, MCC_0, MCC_1, MCC_2);
                thisR(mi).m = m;
            end
            
            % Algorithm #3 - Original spike as input with Lasso
            tic
            tStar = tic;
            [B_3,FitInfo_3] = lassoglm(P_0, double(thisTrial_c),'binomial', 'Lambda', lambdaPool, 'CV', 10);
            idxLambdaMinDeviance_3 = FitInfo_3.IndexMinDeviance;
            B0_3 = FitInfo_3.Intercept(idxLambdaMinDeviance_3);
            coef_3 = [B0_3; B_3(:,idxLambdaMinDeviance_3)];
            prob_3 = glmval(coef_3, P_0_test, 'logit');
            predict_3 = double(prob_3>=0.5);
            CM_3 = confusionmat(testingTarget, predict_3);
            MCC_3 = mcc(CM_3);
            tElapsed_3 = toc(tStar);
            thisR(mi).prob_3 = prob_3; thisR(mi).CM_3 = CM_3; thisR(mi).MCC_3 = MCC_3; thisR(mi).tElapsed_3 = tElapsed_3;
            
            % Algorithm #4 - PCA features as input with Lasso
            tic
            tStar = tic;
            [B_4,FitInfo_4] = lassoglm(P_score, double(thisTrial_c),'binomial', 'Lambda', lambdaPool, 'CV', 10);
            idxLambdaMinDeviance_4 = FitInfo_4.IndexMinDeviance;
            B0_4 = FitInfo_4.Intercept(idxLambdaMinDeviance_4);
            coef_4 = [B0_4; B_4(:,idxLambdaMinDeviance_4)];
            prob_4 = glmval(coef_4, P_score_test, 'logit');
            predict_4 = double(prob_4>=0.5);
            CM_4 = confusionmat(testingTarget, predict_4);
            MCC_4 = mcc(CM_4);
            tElapsed_4 = toc(tStar);
            thisR(mi).prob_4 = prob_4; thisR(mi).CM_4 = CM_4; thisR(mi).MCC_4 = MCC_4; thisR(mi).tElapsed_4 = tElapsed_4;
                   
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            fprintf('MCC lasso:%1.2f; MCC LR:%1.2f; MCC NB:%1.2f; MCC lasso_0:%1.2f; MCC lasso_pca:%1.2f\n', bestMCC_0, bestMCC_1, bestMCC_2, MCC_3, MCC_4);
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
            
        end
        function R = runAllTrials(obj,IfNotCtrl,varargin)
            IfPlot = process_options(varargin,'IfPlot',0);
            SpikeTensor=obj.getSpikeTensor;
            if obj.par
                if IfPlot
                    warning('Plotting not supported for parallel')
                end
                parfor ti = 1:obj.Num_Trial
                    R_sepTrial{ti} = obj.runTrial(ti,IfNotCtrl,SpikeTensor);
                end
            else
                for ti = 1:obj.Num_Trial
                    R_sepTrial{ti} = obj.runTrial(ti,IfNotCtrl,SpikeTensor,'IfPlot',IfPlot);
                end
            end
            %Put R back in the original format
            for ti = 1:obj.Num_Trial
                % prepare spatio-temporal patterns for classification
                for mi = 1:length(obj.m_all)
                    R(ti,mi) = R_sepTrial{ti}(mi);
                end
            end
            
        end
        function MDfit = run(obj,varargin)
            if obj.par
                poolOb = obj.setupPool;
            end
            tStart = tic;
            [IfPlot] = process_options(varargin,'IfPlot',0);
            IfNotCtrl = 1;
            obj.R_exp = obj.runAllTrials(IfNotCtrl,'IfPlot',IfPlot); % Experiment Group
            % Caution Here !!!!!!!!!!!! Only for sample response case
            %             IfNotCtrl = 0;
            %             obj.R_ctrl = obj.runAllTrials(IfNotCtrl,'IfPlot',IfPlot);% Control Group
            tFit = toc(tStart);
            obj.tFit = tFit;
            MDfit = obj;
            save(obj.oF,'MDfit','-v7.3')
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

