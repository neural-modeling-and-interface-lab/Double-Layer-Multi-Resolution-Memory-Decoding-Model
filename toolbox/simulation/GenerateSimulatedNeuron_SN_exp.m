function [neuronSpikes, p_n1_c1] = GenerateSimulatedNeuron_SN_exp(mu, sigma, p0, nTrials, t, L, intense)

p1_c1 = normpdf(t,mu,sigma)*intense; % first peak
p_n1_c1 = p0+p1_c1;

neuronSpikes = zeros(L,nTrials);
for i = 1:nTrials % Keep the same pdf for all trials
    r = rand(L,1);
    neuronSpikes(:,i)=(r<p_n1_c1);
end