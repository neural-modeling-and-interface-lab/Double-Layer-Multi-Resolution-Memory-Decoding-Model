function [neuronSpikes, p_n1_c1] = GenerateSimulatedNeuron_SN_ctrl(~, ~, p0, nTrials, ~, L, ~)
randomSeeds = 0;
p_n1_c1 = p0;

neuronSpikes = zeros(L,nTrials);
for i = 1:nTrials % Keep the same pdf for all trials
    r = rand(L,1);
    neuronSpikes(:,i)=(r<p_n1_c1);
end