function [maxind, minind] = peakfinder2(X,eta,threshold)
%% This function calculates the max and min for the series x, with threshold
% threshold and spacing spacing (in units of x)

ploton = 0;

spacing = round(threshold / (X(2) - X(1)));

maxind = [];
minind = [];

for i = 1:length(eta)
    
    first = max(1,i - spacing);
    last = min(length(eta),i+spacing);
    
    compmax = max(eta(first:last));
    if eta(i) >= compmax
        maxind = [maxind i];
    end
    
    compmin = min(eta(first:last));
    if eta(i) <= compmin
        minind = [minind i];
    end
    
end


%%

% Sort out in between points
for i = 1:length(minind)-1
    
    p = (maxind > minind(i)).*(maxind < minind(i+1));
    p = sum(p(:));
    
    if p == 0
        
        
        [~, loc] = max(eta(minind(i)+1:minind(i+1)-1));
        
        maxind = sort([maxind loc + minind(i)]);
        
    end
    
end

for i = 1:length(maxind)-1
    
    p = (minind > maxind(i)).*(minind < maxind(i+1));
    p = sum(p(:));
    
    if p == 0
        
        [~, loc] = min(eta(maxind(i)+1:maxind(i+1)-1));
        
         minind = sort([minind loc + maxind(i)]);
    end
    
end



if ploton
    
    plot(X,eta,'k');
    hold on
    scatter(X(maxind),eta(maxind),'r','filled');
    scatter(X(minind),eta(minind),'b','filled');
    xlim([0 1000])
    grid on
    box on
    ylabel('SSH (m)')
    set(gca,'layer','top','fontname','helvetica','fontsize',12)
    pos = [8 4];
    set(gcf,'windowstyle','normal','position',[0 0 pos],'paperposition',[0 0 pos],'papersize',pos,'units','inches');
    set(gcf,'windowstyle','normal','position',[0 0 pos],'paperposition',[0 0 pos],'papersize',pos,'units','inches');
    saveas(gcf,'examp.pdf')
    hold off
    
    
    
end

end