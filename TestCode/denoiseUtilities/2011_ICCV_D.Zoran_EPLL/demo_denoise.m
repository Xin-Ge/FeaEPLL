clear
patchSize = 8;

% load image
I = double(rgb2gray(imread('160068.jpg')))/255;	
I = im2double(imread('08.png'));	

% add noise
noiseSD = 25/255;
noiseI = I + noiseSD*randn(size(I));
excludeList = [];

% set up prior
LogLFunc = [];
load GSModel_8x8_200_2M_noDC_zeromean.mat
prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);

%%
tic
% add 64 and 128 for high noise
[cleanI,psnrs,~] = EPLLhalfQuadraticSplit(noiseI,patchSize^2/noiseSD^2,patchSize,(1/noiseSD^2)*[1 4 8 16 32],1,prior,I,LogLFunc);
toc

% output result
figure(1);
imshow(I); title('Original');
figure(2);
imshow(noiseI); title('Corrupted Image');
figure();
imshow(uint8(cleanI*255)); title(sprintf('Restored Image, PSNR = %.6f, SSIM = %.6f', psnr(cleanI,I), ssim(cleanI,I)));
% fprintf('PSNR is:%f\n',20*log10(1/std2(cleanI-I)));
% fprintf('PSNR is:%f\n',psnr(cleanI, I));

