function rasterplot(Y)
%% Raster plot

[N, L] = size(Y);

BinSize = 0.002;
for i=1:N
    t = find(Y(i,:));
    hold on;
    if ~isempty(t)
        %             plot(t, i*ones(size(t)),'k.'); % plot dots
        for j = 1:length(t)
            plot([t(j) t(j)]*BinSize-L*BinSize/2, [i-.75 i-.25],'k','LineWidth',1.2);
        end
    else
    end
end
axis([-L*BinSize/2 L*BinSize/2 -0.5 N+0.5]); set(gca,'TickDir','out')
box on;