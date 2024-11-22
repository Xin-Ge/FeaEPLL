function [cleanI, evaluate] = interface_EPLL2048_NoisyDC(noiseI,clearI,sigma)
s = tic;

noiseI = noiseI ./ 255;
clearI = clearI ./ 255;
sigma  = sigma  ./ 255;

% set up prior
load('gradGmm_gray_single_2048_06_joint_v2_Gk2.mat')

% add 64 and 128 for high noise
[cleanI, ~] = Grad_EPLL_sp_6_5(noiseI, sigma, 0.05, prior, 4, clearI);

cleanI = cleanI .* 255;
clearI = clearI .* 255;
sigma  = sigma  .* 255;

psnrValue = psnr(single(cleanI)/255, single(clearI)/255);
ssimValue = ssim(single(cleanI)/255, single(clearI)/255);
evaluate = struct('psnr', psnrValue, 'ssim', ssimValue);

fprintf('EPLL with PN by removing clean DC: noiselevel = %.2f, time = %.2f s, psnr = %.2f ssim = %.4f \n', ...
    sigma, toc(s), psnrValue, ssimValue);
end