clear
% Initialize stuff
repo_path = '/Users/chorvat/Dropbox (Brown)/Research Projects/Active/WIFF-Model/'

addpath([repo_path '/Misc/']); 
%% Initialization

Domainwidth = 1e4; % meters
% Discretize the input
eps = 1e-8;

% The discretization is equal to the mean floe size spacing divided by two
dx = 1;

% This creates the vector, which has length equal to Domainwidth
X = dx:dx:Domainwidth; % Domain discretization

maxcounts = 1000; % Max number of interations before quitting
errtol = 6.5e-4;  % mean absolute error in successive histograms

% Floe size discretization
redges = [6.65000000e-02,   5.31030847e+00,   1.42865861e+01,   2.90576686e+01,5.24122136e+01,   8.78691405e+01,   1.39518470e+02,   2.11635752e+02, 3.08037274e+02,   4.31203059e+02,   5.81277225e+02,   7.55141047e+02, 9.45812834e+02]; 
rcenters = (redges(1:end-1) + redges(2:end))/2; 
dR = diff(redges);

load([repo_path '/Misc/NN_params.mat'],'f'); % frequencies from WW3
df = f .* (sqrt(1.1) - sqrt(1/1.1)); 

gravity = 9.80616; 

Lambda = gravity./(2*pi*f.^2); 
bandwidth = 10;
epscrit = 3e-5;

%% Input to standalone code

% An example wave/H/C code that is from CICE 
WSPEC = [2.322543878108263e-4,1.589981839060783e-3,8.112276904284954e-3,3.017734736204147e-2,8.668871968984604e-2,0.281472265720367,0.749001085758209,1.08791637420654,1.10219252109528,0.772392153739929,0.559300124645233,0.394673168659210,0.226335361599922,0.133877992630005,9.566487371921539e-2,6.856952607631683e-2,4.777402803301811e-2,3.151062130928040e-2,1.904737576842308e-2,1.082132477313280e-2,5.622150376439095e-3,2.765464130789042e-3,1.271661953069270e-3,5.112921935506165e-4,3.174721496179700e-4]';
H = 2.02407119175920; 
C = 5.342104974855429e-2;

% No attenuation because that is handled by CICE/WW3
coeffs = sqrt(2*WSPEC .* abs(df));

dX = [];

err = 100000;
FHold = zeros(1,length(rcenters));


%%
counter = 0; 

% Compute A until successive histograms are close. 
while (counter < maxcounts && (err > errtol)) || counter < 2
    
    counter = counter + 1;
    
    % We create a random phase for each fourier component
    % randphase = 2*pi*.5*ones(1,length(WSPEC)); %
    randphase = 2*pi*rand(1,length(WSPEC)); % Random phases
    
    % Construct Sea Surface Height Record
    fourier = cos(bsxfun(@plus,bsxfun(@times,2*pi./Lambda,X),randphase'));
    
    % Sea surface height is obtained by summing the Fourier components
    eta = sum(coeffs.*fourier,1);
    
    % We use the peakfinder code to find peaks of the spectrum
    [maxloc, minloc] = peakfinder2(X,eta,bandwidth);
    
    % This now is the array where pairs of values are maxes and mins
    extremelocs = sort([minloc maxloc]);
    
    extremelocs = unique(extremelocs);
    
    % The values in space of each extreme point
    extremex = X(extremelocs);
    % The SSH values of each extreme point
    extremeeta = eta(extremelocs);
    
    
    %%
    % First increment
    dx1 = extremex(2:end-1) - extremex(1:end-2);
    % Second increment
    dx2 = extremex(3:end) - extremex(2:end-1);
    
    % Numerator of strain
    D2Y = 2*(extremeeta(1:end-2).*dx2 - extremeeta(2:end-1).*(dx2 + dx1) + extremeeta(3:end).*dx1);
    
    % Denominator of strain
    D2X = dx1 .*dx2 .* (dx1 + dx2);
    
    if isempty(D2X)
        D2X = Inf;
        D2Y = 0;
    end
    
    strain = 0*extremex; 
    strain(2:end-1) = .5*(bsxfun(@times,H,abs(D2Y./D2X)));  % Calculate strain magnitude
    % strain is now a NFRAC by H size array
    
    % Those values exceeding strain threshold
    abovelocs = strain > epscrit;
    
    % For each H, we pick out all of the locations that will lead to
    % fracture, and put that in Xlocs. Then we take the distance between
    % them and call that the fracture lengths, and store those in a new
    % array
    if sum(abovelocs(:)) ~= 0
        
        % The extreme X vales for ice of this thickness
        Xlocs = extremex(abovelocs);
        % This is now the vector or fracture lengths implied by the sea
        % surface height field.
        dX = [dX diff(Xlocs)/2];
        
    end
    
    [fracture_hist,~]  = histcounts(dX,redges) ;
    
    fracture_hist(end) = fracture_hist(end) + sum(dX>redges(end)); 

    
    if sum(fracture_hist) > 0
        
        % Normalize so that sum(fracture_hist) = 1;
        fracture_hist = (fracture_hist .* rcenters)/sum(fracture_hist.*rcenters);
        
    end
    
    % Mean absolute error
    err = mean(abs(fracture_hist - FHold));
    FHold = fracture_hist;
    
end

%% Rebin to Lettie's bounds

close all
figure(1)
subplot(111)

yyaxis left; 
set(gca,'ycolor','k'); 
semilogx(Lambda,WSPEC,'color','k','linewidth',1); 
ylabel('Wave Spectrum'); 
grid on; box on; 
hold on
xlim([rcenters(1) rcenters(end)]); 


yyaxis right; set(gca,'ycolor','r'); 
semilogx(rcenters,fracture_hist,'r','linewidth',1); 
xlabel('Size/Wavelength (m)');
ylabel('A_idr_i'); 
grid on; box on; 
xlim([rcenters(1) rcenters(end)]); 

legend('Wave Spectrum','Histogram'); 


