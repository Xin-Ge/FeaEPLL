function x = Denoise_FeaEPLL(ynoise, noiseSD, yorig)
%% %%%%%%%%%%%%%%%%%%%%%%%‘§¥¶¿Ì%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [];
TemCodepath = genpath(fullfile(pwd, 'denoiseUtilities', 'FeaEPLL'));
addpath(TemCodepath)

% load GMM model
if size(ynoise,3) == 1
    load('gradGmm_gray_single_2048_06_joint_Gk2.mat')
else
    load('gradGmm_color_single_1024_05_joint_Gk2.mat')
end

ynoise = single(ynoise);
yorig  = single(yorig);
[x,~] = FeaEPLL_denoising(ynoise, noiseSD, 0.05, prior, 4, yorig, 1);

rmpath(TemCodepath);
end