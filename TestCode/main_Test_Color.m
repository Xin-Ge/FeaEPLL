clear;
close all;% clc;

%% %%%%%%%%%%%%%%%%%%%%%% Set parameters %%%%%%%%%%%%%%%%%%%%%%%%%%
ColorType       = 'Color';
methodSelect    = [2];
dataSelect      = {'Set12', 'BSD68', 'Urban100'};
noiseLeve       = [15 25 50 75 100];   % Set Noise Level:0~100
methodMainCode  = fullfile(pwd, 'denoiseUtilities');
oriDataDir      = fullfile(pwd, 'testsets', 'original');
temDataDir      = fullfile(pwd, 'testsets', 'noisy', ColorType);
resultDir       = fullfile(pwd, 'results', 'denoising', ColorType);

addpath(methodMainCode);

%% %%%%%%%%%%%%%%%%%%%%%%% Test methods %%%%%%%%%%%%%%%%%%%%%%%%%%%
CompareMethod{1}  = 'EPLL';    % 2011 ICCV D.Zoran
% From Learning Models of Natural Image Patches to Whole Image Restoration
CompareMethod{2}  = 'FeaEPLL';

%% %%%%%%%%%%%%%%% prepare test images %%%%%%%%%%%%%%
ext        = {'*.jpg','*.bmp','*.tif','*.png','*.gif','*.jpeg'};
filelist   = [];
for nF = 1:numel(dataSelect)
    readFolder = fullfile(oriDataDir, dataSelect{nF});
    for ne = 1:numel(ext)
        filelist = cat(1,filelist,dir(fullfile(readFolder, ext{ne})));
    end
end

for nN = 1:length(noiseLeve)
    for nF = 1:numel(dataSelect)
        noisyFolder = fullfile(temDataDir, sprintf('sigma_%d',noiseLeve(nN)), dataSelect{nF});
        if ~exist(noisyFolder, 'dir')
            mkdir(noisyFolder);
        end
        for nM = 1:length(methodSelect)
            resultFolder = fullfile(resultDir, CompareMethod{methodSelect(nM)}, sprintf('sigma_%d',noiseLeve(nN)), dataSelect{nF});
            if ~exist(resultFolder, 'dir')
                mkdir(resultFolder);
            end
        end
    end
end

ssim_noisy  = zeros([numel(filelist), length(noiseLeve), length(methodSelect)]);
psnr_noisy  = zeros([numel(filelist), length(noiseLeve), length(methodSelect)]);
ssim_recon  = zeros([numel(filelist), length(noiseLeve), length(methodSelect)]);
psnr_recon  = zeros([numel(filelist), length(noiseLeve), length(methodSelect)]);
for nM = 1:length(methodSelect)                                             % Main Denoise Code
    fprintf('Test method: %s\n', CompareMethod{methodSelect(nM)});
    for nN = 1:length(noiseLeve)
        sigma = noiseLeve(nN);
        fprintf('Test the noiselevel = %d\n', sigma);
        for nI = 1:numel(filelist)
            fprintf('Test the image %d/%d: %s\n', nI, numel(filelist), filelist(nI).name);
            
            currData  = filelist(nI).folder(find(filelist(nI).folder == filesep, 1, 'last')+1 : end);
            currName  = filelist(nI).name(1 : find(filelist(nI).name == '.', 1, 'last')-1);
            noisyFile = fullfile(temDataDir, sprintf('sigma_%d',sigma), currData, sprintf('%s.mat', currName));
            if exist(noisyFile, 'file')
                load(noisyFile);
            else
                yorig  = im2double(imread(fullfile(filelist(nI).folder, filelist(nI).name)));                            % Read Original Image
                [~,~,chNum] = size(yorig);
                if chNum == 1
                    continue;
                end
                ynoisy = yorig + (sigma/255) * randn(size(yorig), 'like', yorig);           % Add Noise
                save(noisyFile, 'yorig', 'ynoisy', '-v7.3');
                imwrite(im2uint8(ynoisy), fullfile(temDataDir, sprintf('sigma_%d',sigma), currData, sprintf('%s.png', currName)));
            end
            
            resultFolder = fullfile(resultDir, CompareMethod{methodSelect(nM)}, sprintf('sigma_%d',noiseLeve(nN)), currData);
            reconFile = fullfile(resultFolder, sprintf('%s.mat', currName));
            if exist(reconFile, 'file')
                load(reconFile);
                ssim_noisy(nI,nN,nM) = ssim(ynoisy, yorig);
                psnr_noisy(nI,nN,nM) = psnr(ynoisy, yorig);
                ssim_recon(nI,nN,nM) = ssim(xlata,  yorig);
                psnr_recon(nI,nN,nM) = psnr(xlata,  yorig);
            else
                s = tic;
                xlata = feval(['Denoise_',CompareMethod{methodSelect(nM)}], ynoisy, sigma/255, yorig);
                [Row, Col] = size(yorig);
                fprintf('Time taken for image of size %d x %d is %.3f s\n', Row, Col, toc(s));
                
                %% calculate evaluation index value
                if ~isempty(xlata)
                    ssim_noisy(nI,nN,nM) = ssim(single(ynoisy), single(yorig));
                    psnr_noisy(nI,nN,nM) = psnr(single(ynoisy), single(yorig));
                    ssim_recon(nI,nN,nM) = ssim(single(xlata),  single(yorig));
                    psnr_recon(nI,nN,nM) = psnr(single(xlata),  single(yorig));

                    save(reconFile, 'xlata', '-v7.3');
                    imwrite(im2uint8(xlata), ...
                        fullfile(resultFolder, sprintf('%s_ssim%.4f_psnr%.2f.png', currName, ssim_recon(nI,nN,nM), psnr_recon(nI,nN,nM))));
                end
            end
        end
    end
end
