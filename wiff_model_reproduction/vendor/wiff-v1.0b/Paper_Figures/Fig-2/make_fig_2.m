%% Make_Fig_1.m
% Creates the first figure in Horvat and Roach (2021). 
% This code starts as if drive_plotting and training_preamble have both
% been called previously. 


figure(2)
clf; clear Ax; 

%% Group error by input energy/concentration/thickness

[NC,~,bX] = histcounts(conc,Cbins);  
errC = accumarray(bX,perc_error,[nbins_C 1],@nanmedian); 
varC = accumarray(bX,perc_error,[nbins_C 1],@iqr);

[NH,~,bX] = histcounts(thick,Hbins);  
errH = accumarray(bX,perc_error,[nbins_H 1],@nanmedian); 
varH = accumarray(bX,perc_error,[nbins_H 1],@iqr);


[NE,~,bX] = histcounts(E,Ebins);  
errE = accumarray(bX,perc_error,[nbins_E 1],@nanmedian); 
varE = accumarray(bX,perc_error,[nbins_E 1],@iqr);


[NR,~,bX] = histcounts(mean_FS_true,Rbins);  
errR = accumarray(bX,perc_error,[nbins_R 1],@nanmedian); 
varR = accumarray(bX,perc_error,[nbins_R 1],@iqr);
errR(NR < 1) = nan; 
varR(NR < 1) = nan; 


%%
horvat_colors; 

Ax{1} = subplot(321);

yyaxis left; set(gca,'ycolor','k'); 
histogram(conc,Cbins,'normalization','probability','facecolor',clabs(1,:),'edgecolor','none');
grid on; box on; xlim(Clims);  
xlabel('Ice Concentration','interpreter','latex'); 
ylabel('N(C)dC','interpreter','latex','color',clabs(1,:)); 
xlim(Clims);

yyaxis right; set(gca,'ycolor','k'); 
plot(Cbins(1:end-1),errC,'k','linewidth',1); 
hold on
plot(Cbins(1:end-1),errC+varC/2,'--k','linewidth',0.5); 
plot(Cbins(1:end-1),errC-varC/2,'--k','linewidth',0.5); 
hold off
xlim(Clims); 
ylim([0 15]);
ylabel('SSE (\%)','interpreter','latex'); 

Ax{2} = subplot(322);

yyaxis left; set(gca,'ycolor','k'); 
histogram(thick,Hbins,'normalization','probability','facecolor',clabs(1,:),'edgecolor','none');
grid on; box on; xlim(Hlims); 
ylabel('N(H)dH','interpreter','latex','color',clabs(1,:)); 
xlabel('Ice Thickness','interpreter','latex'); 
xlim(Hlims);

yyaxis right; set(gca,'ycolor','k'); 
plot(Hbins(1:end-1),errH,'k','linewidth',1); 
hold on
plot(Hbins(1:end-1),errH+varH/2,'--k','linewidth',0.5); 
plot(Hbins(1:end-1),errH-varH/2,'--k','linewidth',0.5); 
hold off
xlim(Hlims); ylim([0 15]); 
ylabel('SSE (\%)','interpreter','latex'); 

Ax{3} = subplot(323);

yyaxis left; set(gca,'ycolor','k'); 
histogram(E,Ebins,'normalization','probability','facecolor',clabs(1,:),'edgecolor','none');
grid on; box on; xlim(Elims); 
xlabel('Log$_{10}$ Wave Energy','interpreter','latex'); 
ylabel('N(E)dE','interpreter','latex','color',clabs(1,:)); 
xlim(Elims);


yyaxis right; set(gca,'ycolor','k'); 
plot(Ebins(1:end-1),errE,'k','linewidth',1); 
hold on
plot(Ebins(1:end-1),errE+varE/2,'--k','linewidth',0.5); 
plot(Ebins(1:end-1),errE-varE/2,'--k','linewidth',0.5); 
hold off
xlim(Elims); 
ylim([0 15]); 
ylabel('SSE (\%)','interpreter','latex'); 

Ax{4} = subplot(324); 

yyaxis left; set(gca,'ycolor','k'); 
histogram(mean_FS_true,Rbins,'normalization','probability','facecolor',clabs(1,:),'edgecolor','none');
grid on; box on; xlim(Rlims); 
xlabel('$\overline{R}$','interpreter','latex'); 
ylabel('N(R)dR','interpreter','latex','color',clabs(1,:)); 
set(gca,'xscale','log'); 
xlim(Rlims); 

yyaxis right; set(gca,'ycolor','k'); 
plot(Rbins(1:end-1),errR,'k','linewidth',1); 
hold on
plot(Rbins(1:end-1),errR+varR/2,'--k','linewidth',0.5); 
plot(Rbins(1:end-1),errR-varR/2,'--k','linewidth',0.5); 
hold off
set(gca,'xscale','log'); 
xlim(Rlims); 
ylim([0 25]); 
ylabel('SSE (\%)','interpreter','latex'); 

%%


Ax{5} = subplot(313);

varvals = nanstd(Y_pred - Y_true,[],1); 
semilogx(rcent,nanmean(Y_true,1),'-k','linewidth',2); 
hold on
semilogx(rcent,nanmean(Y_pred,1),'-r','linewidth',1);
scatter(rcent,0*nanmean(Y_pred,1),50,'+','k'); 
semilogx(rcent,nanmean(Y_true,1)+varvals,'--k','linewidth',0.5); 
% 
semilogx(rcent,nanmean(Y_true,1)-varvals,'--k','linewidth',0.5); 
hold off
grid on; box on; 
ylim([0 .5]);
xlim(Rlims); 
xlabel('Floe Size (m)','interpreter','latex'); ylabel('FSD (\o)','interpreter','latex'); 
legend('True','Predicted'); 


%%
letter = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(e)','(c)'};

delete(findall(gcf,'Tag','legtag'))

for i = 1:length(Ax)
    
    posy = get(Ax{i},'position');

    set(Ax{i},'fontname','helvetica','fontsize',8,'xminortick','on','yminortick','on')
    
    annotation('textbox',[posy(1) posy(2)+posy(4) - .02 .025 .025], ...
        'String',letter{i},'LineStyle','none','FontName','Helvetica', ...
        'FontSize',8,'Tag','legtag');
    
end

pos = [6.5 4]; 
set(gcf,'windowstyle','normal','position',[0 0 pos],'paperposition',[0 0 pos],'papersize',pos,'units','inches','paperunits','inches');
set(gcf,'windowstyle','normal','position',[0 0 pos],'paperposition',[0 0 pos],'papersize',pos,'units','inches','paperunits','inches');
saveas(gcf,[figure_save_path '/Fig-2/Fig-2.pdf']); 