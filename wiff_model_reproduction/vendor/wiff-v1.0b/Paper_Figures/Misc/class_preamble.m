clearvars -except *_path

load([repo_path '/Misc/NN_params.mat']);
load(classifier_path);

class_threshold = 0.54; 

Y_true = Y_true > class_threshold; 
Y_pred = Y_pred > class_threshold; 

%% Basic Setup

truepos  = (Y_true == 1 & Y_pred == 1); 
trueneg  = (Y_true == 0 & Y_pred == 0); 
falsepos = (Y_true == 0 & Y_pred == 1); 
falseneg = (Y_true == 1 & Y_pred == 0); 

df = f' * (sqrt(1.1) - sqrt(1/1.1));

conc = X_test(:,end);
thick = X_test(:,end-1);
spec = X_test(:,1:25);
E = log10(sum(bsxfun(@times,df,spec),2));

nbins_C = 48;
nbins_H = 58;
nbins_E = 50;
nbins_R = 45; 

Hbins = [linspace(-.01,4,nbins_H) Inf]';
Ebins = [linspace(-3.3,1,nbins_E) Inf]';
Cbins = [linspace(-.01,1.01,nbins_C) Inf]';
Rbins = [logspace(log10(rcent(1)),log10(rcent(end)),nbins_R) Inf]'; 

Hlims = [0 3];
Clims = [0 1];
Elims = [-3 1];
Rlims = [rcent(1) rcent(end)]; 
