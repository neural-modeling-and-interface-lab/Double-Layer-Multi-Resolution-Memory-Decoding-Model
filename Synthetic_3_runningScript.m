%% This script is used for running the proposed Double-Layer Multi-Resolution Memory Decoding Model
% modeled resolution [5:25, 50:5:150] case #1-4
% plus new added resolution [0:4] case #5
% Project: RAM USC
% Author: Xiwei She
% Date: 2020-11-09

%% Uncomment/comment corresponding cases you want to check

% Note by Xiwei: backup previous data/results before run this
% Be care of overwritting

%% NestCV  DLMDM - Synthetic case #1 SN&LS
% clear; clc;
% addpath(genpath('toolbox'));
% addpath('..');  %adds MemoryDecode Class and process_options
% 
% % Settings
% Patitions = 1:10; % 10-fold nested cross-validation
% m_all = [5:25, 50:5:150];
% Num_Trial = 8;
% 
% nCategories = 2;
% Category = 'A';
% Subject = 'SingleNeuron';
% runCase = 'SN&LS4';
% 
% for pa = 1
%     for currentPartition = Patitions
%         printStr1 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' Begine!'];
%         disp(printStr1);
%         parMD = DLMDM_Simulated_singleNeuron(Subject, Category, currentPartition, nCategories, runCase, 'par',1,'m_all',m_all,'Num_Trial',Num_Trial);
%         parMD_R = parMD.run;
%         printStr2 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' End!'];
%         disp(printStr2);
%     end
% end

%% NestCV  DLMDM - Synthetic case #2 SN&HS
% clear; clc;
% addpath(genpath('toolbox'));
% addpath('..');  %adds MemoryDecode Class and process_options
% 
% % Settings
% Patitions = 1:10; % 10-fold nested cross-validation
% m_all = [5:25, 50:5:150];
% Num_Trial = 8;
% 
% nCategories = 2;
% Category = 'A';
% Subject = 'SingleNeuron';
% runCase = 'SN&HS4';
% 
% for pa = 1
%     for currentPartition = Patitions
%         printStr1 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' Begine!'];
%         disp(printStr1);
%         parMD = DLMDM_Simulated_singleNeuron(Subject, Category, currentPartition, nCategories, runCase, 'par',1,'m_all',m_all,'Num_Trial',Num_Trial);
%         parMD_R = parMD.run;
%         printStr2 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' End!'];
%         disp(printStr2);
%     end
% end


%% NestCV  DLMDM - Synthetic case #3 CN&CS
% clear; clc;
% addpath(genpath('toolbox'));
% addpath('..');  %adds MemoryDecode Class and process_options
% 
% % Settings
% Patitions = 1:10; % 10-fold nested cross-validation
% m_all = [5:25, 50:5:150];
% Num_Trial = 8;
% 
% nCategories = 2;
% Category = 'A';
% Subject = 'CombinedNeuron';
% runCase = 'CN&CS4';
% 
% for pa = 1
%     for currentPartition = Patitions
%         printStr1 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' Begine!'];
%         disp(printStr1);
%         parMD = DLMDM_Simulated_combinedNeuron(Subject, Category, currentPartition, nCategories, runCase, 'par',1,'m_all',m_all,'Num_Trial',Num_Trial);
%         parMD_R = parMD.run;
%         printStr2 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' End!'];
%         disp(printStr2);
%     end
% end


%% NestCV  DLMDM - Synthetic case #4 population
% clear; clc;
% addpath(genpath('toolbox'));
% addpath('..');  %adds MemoryDecode Class and process_options
% 
% % Settings
% Patitions = 1:10; % 10-fold nested cross-validation
% m_all = [0:25, 50:5:150];
% Num_Trial = 8;
% 
% nNeuron = 30;
% nCategories = 5;
% nTrials = 500;
% runCase = [mat2str(nNeuron), 'N&', mat2str(nTrials), 'T&', mat2str(nCategories),'C']; 
% for ca = 1:5 % 1:5
%     switch ca
%         case 1
%             currentCategory = 'A';
%         case 2
%             currentCategory = 'B';
%         case 3
%             currentCategory = 'C';
%         case 4
%             currentCategory = 'D';
%         case 5
%             currentCategory = 'E';
%     end
%     for currentPartition = Patitions
%         printStr1 = [runCase, ' Category', currentCategory, ' Begine!'];
%         disp(printStr1);
%         parMD = DLMDM_Simulated_realistic(nNeuron, nCategories, currentPartition, currentCategory, runCase, 'par',1,'m_all',m_all,'Num_Trial',Num_Trial);
%         parMD_R = parMD.run;
%         printStr2 = [runCase, ' Category', currentCategory,  ' End!'];
%         disp(printStr2);
%     end
% end

%% NestCV  DLMDM - Synthetic case #5 additional min resolution
% clear; clc;
% addpath(genpath('toolbox'));
% addpath('..');  %adds MemoryDecode Class and process_options
% 
% % Settings
% Patitions = 1:10; % 10-fold nested cross-validation
% m_all = 0:4;
% Num_Trial = 8;
% 
% nCategories = 2;
% Category = 'A';
% Subject = 'CombinedNeuron';
% runCase = 'CN&CS4';
% 
% for pa = 1
%     for currentPartition = Patitions
%         printStr1 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' Begine!'];
%         disp(printStr1);
%         parMD = DLMDM_Simulated_combinedNeuron_minRes(Subject, Category, currentPartition, nCategories, runCase, 'par',1,'m_all',m_all,'Num_Trial',Num_Trial);
%         parMD_R = parMD.run;
%         printStr2 = [Subject, '&Category', Category, '_part#', mat2str(currentPartition), ' End!'];
%         disp(printStr2);
%     end
% end

%% Case #6 Algorithm Comparison - Synthetic
% clear; clc;
% addpath(genpath('toolbox'));
% addpath('..');  %adds MemoryDecode Class and process_options
% 
% % Settings
% Patitions = 1:10; % 1:10
% m_all = [0:25, 50:5:150];
% Num_Trial = 4;
% 
% numCategory = [2, 5, 1]; % '1' corresponding to 30N&500T&5C
% 
% for i = numCategory
%     for currentPartition = Patitions
%         Category = [mat2str(i), 'C'];
%         printStr1 = [Category, ' part#', mat2str(currentPartition), ' Begine!'];
%         disp(printStr1);
%         parMD = MemoryDecode_algorithmComparison_simulation(Category, currentPartition, 'par',1,'m_all',m_all,'Num_Trial',Num_Trial);
%         parMD_R = parMD.run;
%         printStr2 = [Category, ' End!'];
%         disp(printStr2);
%     end
% end