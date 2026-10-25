clearvars -except *_path

load([repo_path '/Misc/NN_params.mat']);
load(validation_path);

%% Basic Setup

df = f' * (sqrt(1.1) - sqrt(1/1.1));

conc = X_test(:,end);
thick = X_test(:,end-1);
spec = X_test(:,1:25);
E = log10(sum(bsxfun(@times,df,spec),2));

nbins_C = 48;
nbins_H = 58;
nbins_E = 50;
nbins_R = 45; 

Hbins = [linspace(0,4,nbins_H) Inf]';
Ebins = [linspace(-3.3,1,nbins_E) Inf]';
Cbins = [linspace(0,1,nbins_C) Inf]';
Rbins = [logspace(log10(rcent(1)),log10(rcent(end)),nbins_R) Inf]'; 

Hlims = [0 3];
Clims = [0 1];
Elims = [-3 1];
Rlims = [rcent(2) .8*rcent(end)]; 

%% Some Derived Quantities

mean_FS_true = squeeze(nansum(bsxfun(@times,rcent,Y_true),2));
mean_FS_pred = squeeze(nansum(bsxfun(@times,rcent,Y_pred),2));
perc_error = 100*abs(mean_FS_true - mean_FS_pred)./ (.5*(mean_FS_true + mean_FS_pred));

