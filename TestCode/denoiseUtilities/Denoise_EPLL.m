function x = Denoise_GMM(ynoise, noiseSD, yorig)
%% %%%%%%%%%%%%%%%%%%%%%%%‘§¥¶¿Ì%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [];
TemCodepath = genpath(fullfile(pwd, 'denoiseUtilities', '2011_ICCV_D.Zoran_EPLL'));
addpath(TemCodepath)

if size(ynoise,3) == 3
    warning('EPLL donot support color image'); 
    return;
end

% load GMM model
load('GSModel_8x8_200_2M_noDC_zeromean.mat');

% uncomment this line if you want the total cost calculated
% LogLFunc = @(Z) GMMLogL(Z,GS);
patchSize = 8;
% initialize prior function handle
excludeList = [];
prior  = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);

% comment this line if you want the total cost calculated
LogLFunc = [];

lamda     = 1*(patchSize/noiseSD)^2;
betas     = (1/noiseSD)^2*[1 4 8 16 32];

[x,~,~] = EPLLhalfQuadraticSplit(ynoise,lamda,patchSize,betas,1,prior,yorig,LogLFunc);

rmpath(TemCodepath);
end