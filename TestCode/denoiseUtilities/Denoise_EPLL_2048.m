function x = Denoise_EPLL_2048(ynoise, noiseSD, yorig)
%% %%%%%%%%%%%%%%%%%%%%%%%‘§¥¶¿Ì%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [];
TemCodepath = genpath(fullfile(pwd, 'denoiseUtilities', 'EPLL_2048'));
addpath(TemCodepath)

if size(ynoise,3) == 3
    warning('EPLL_2048 donot support color image'); 
    return;
end

% load GMM model
load('Gmm_gray_single_2048_08_v2_Gk1.mat')

[x,~] = Grad_EPLL_sp(ynoise, noiseSD, [1 4 8 16 32], prior, yorig, 1);

rmpath(TemCodepath);
end