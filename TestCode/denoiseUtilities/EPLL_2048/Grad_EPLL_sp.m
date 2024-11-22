function [cleanI, psnrout] = Grad_EPLL_xnnnn(noisyI, noiseSD, betas, prior, groundI, gpuid)

useGPU = true;
if useGPU
    try
        % note: first run on GPU always takes longer than subsequent runs
        gpu = gpuDevice(gpuid); if ~gpu.DeviceSupported, error; end %#ok
        fprintf('Using GPU[%d] ''%s'' (free memory: %.0fMB)\n', gpu.Index, gpu.Name, gpu.FreeMemory/1024^2)
        feature('GpuAllocPoolSizeKb', intmax('int32'));
    catch
        useGPU  = false;
    end
end

if useGPU; noisyI = gpuArray(noisyI); end

if numel(prior.patchInfo.patchSize) == 1
    PSize2 = prior.patchInfo.patchSize.^2;
else
    PSize2 = prod(prior.patchInfo.patchSize);
end
chNum    = size(noisyI,3);
numGk    = numel(prior.patchInfo.Gk);

noisygs  = im2grads(noisyI, prior.patchInfo.Gk, prior.patchInfo.GkSize);

% update noise level of the noisy image
noiseSDs      = noiseSD .* cat(4, prior.patchInfo.noiseScale{:});
DCJudge       = cat(4, prior.patchInfo.removeDC{:});
noiseBascCov  = cat(4, prior.patchInfo.noiseBascCov{:});

% initialize with the noisy image
intergs = noisygs;
lambda  = PSize2 ./ (noiseSDs.^2);

condProbs = cat(1,prior.gmm(:).conditionalProbability);
condGMMs  = cell(numel(condProbs), 1);

repNum  = zeros(1, 1, 1, numGk);
k = 0;
for ngmm = 1:numel(prior.gmm)
    condProb  = prior.gmm(ngmm).conditionalProbability(:);
    jointProb = prior.gmm(ngmm).jointProbability(:);
    for p = 1:numel(condProb)
        k = k + 1;
        subProb = condProb{p};
        condGMMs{k} = prepareCondProb(prior.gmm(ngmm), jointProb, subProb);
        repNum(subProb(1))  = repNum(subProb(1)) + 1;
    end
end

k = 1;
% go through all values of beta
for nbeta = 1:numel(betas)
    betau  = betas(nbeta) ./ (noiseSDs.^2);
    % Z step
    % Extract Z, all overlapping patches from the current estimate
    noisyPs = im2patches(intergs, prior.patchInfo.patchSize, prior.patchInfo.stride, prior.patchInfo.dilate);
    [PRowNum, PColNum, ~, ~] = size(noisyPs);
    noisyPs = reshape(noisyPs, [PRowNum*PColNum, PSize2, chNum, numGk]);
    meanPs  = mean(noisyPs, 2) .* DCJudge;
    noisyPs = noisyPs - meanPs;
    noisyPs = reshape(noisyPs, [PRowNum, PColNum, PSize2*chNum, numGk]);
    
    % calculate the MAP estimate for Z using the given prior
    cleanPs   = zeros(size(noisyPs), 'like', noisyPs);
    noiseCovs = noiseBascCov ./ betau;
    for ngmm = 1:numel(condGMMs)
        labels   = selectGaussFromGMM(noisyPs, condGMMs{ngmm}, noiseCovs, condProbs{ngmm});
        estmPs   = updatePatchbyGauss(noisyPs, condGMMs{ngmm}, labels, noiseCovs, condProbs{ngmm});
        cleanPs(:,:,:,condProbs{ngmm}(1)) = cleanPs(:,:,:,condProbs{ngmm}(1)) + estmPs;
    end
    cleanPs = cleanPs ./ repNum;
    
    % X step
    % average the pixels in the cleaned patches in Z
    cleanPs = reshape(cleanPs, [PRowNum*PColNum, PSize2, chNum, numGk]);
    cleanPs = cleanPs + meanPs;
    cleanPs = reshape(cleanPs, [PRowNum, PColNum, PSize2*chNum, numGk]);
    priorgs = patches2im(cleanPs, prior.patchInfo.patchSize, prior.patchInfo.stride, prior.patchInfo.dilate, 'average');

    counts  = PSize2;
    
    cleanI  = grads2im(priorgs, prior.patchInfo.Gk, prior.patchInfo.GkSize);

    % calculate the PSNR for this step
    psnrout(k) = psnr(gather(cleanI),groundI);

    % output the result to the console
    fprintf('  noisy PSNR is:%f; updata PSNR:%f\n', psnr(gather(noisyI),groundI), psnrout(k));
    k       = k + 1;

    if nbeta < numel(betas)
        % calculate the current estimate for the clean image
        intergs = noisygs .* (lambda ./ (lambda + betau*counts)) + priorgs .* (betau*counts ./ (lambda+betau*counts));
        noisyI  = grads2im(intergs, prior.patchInfo.Gk, prior.patchInfo.GkSize);
    end
end

if useGPU; cleanI = gather(cleanI); end

% clip values to be between 1 and 0, hardly changes performance
cleanI(cleanI>1)=1;
cleanI(cleanI<0)=0;
end

function gs = im2grads(I, Gks, GSize)

if numel(GSize) == 1
    GSize = [GSize, GSize]; 
end
G  = cat(2, Gks{:});
numGk = numel(Gks);
Ch  = size(I,3);
wGs = im2patches(I, GSize);
[wRowNum, wColNum, wLen] = size(wGs);
wSize2 = wLen/Ch;
wGs = reshape(wGs, [wRowNum, wColNum, wSize2, Ch]);
wGs = permute(wGs, [1,2,4,3]);
wGs = reshape(wGs, [wRowNum*wColNum*Ch, wSize2]);
gs  = wGs * G;
gs  = reshape(gs, [wRowNum, wColNum, Ch, numGk]);
end

function I = grads2im(gs, Gks, GkSize)

if numel(GkSize) == 1
    GkSize = [GkSize, GkSize]; 
end
G = cat(2, Gks{:});
F = inv(G);
wSize2 = prod(GkSize);

[wRowNum, wColNum, Ch, numGk] = size(gs);
gs  = reshape(gs, [wRowNum*wColNum*Ch, numGk]);
wGs = gs * F;
wGs = reshape(wGs, [wRowNum, wColNum, Ch, wSize2]);
wGs = permute(wGs, [1,2,4,3]);
wGs = reshape(wGs, [wRowNum, wColNum, wSize2*Ch]);
I   = patches2im(wGs, GkSize, 1, 1, 'average');
end

function labels = selectGaussFromGMM(noisyPs, GMM, noiseCovs, jointProb)
ZPs    = noisyPs(:,:,:,jointProb);
[PRowNum, PColNum, PSize2, FeNum] = size(ZPs);
PNum   = PRowNum * PColNum;
PLen   = PSize2 * FeNum;
ZPs    = reshape(ZPs, [PNum, PLen])';

noiseCov = [];
for fn = 1:FeNum
    noiseCov = blkdiag(noiseCov, noiseCovs(:,:,:,jointProb(fn)));
end

mus    = GMM.means;
mixwei = GMM.mixweights;
covs   = GMM.covs;

[U,p] = chol(covs(:,:,1)+noiseCov);
if p ~= 0; error('ERROR: Sigma is not PD.'); end
Q = inv(U') * (ZPs - mus(:,1));
q = dot(Q, Q, 1);  % quadratic term (M distance)
c = PSize2*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
maxZPGV = log(mixwei(1))-(c+q)/2;
labels  = 1;
for Ci = 2:size(mus, 2)
    [U,p] = chol(covs(:,:,Ci)+noiseCov);
    if p ~= 0; error('ERROR: Sigma is not PD.'); end
    Q = inv(U') * (ZPs - mus(:,Ci));
    q = dot(Q, Q, 1);  % quadratic term (M distance)
    c = PSize2*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
    CZPGV = log(mixwei(Ci))-(c+q)/2;
    CMInd = CZPGV > maxZPGV;
    labels  = Ci .* CMInd + labels .* ~CMInd;
    maxZPGV = CZPGV .* CMInd + maxZPGV .* ~CMInd;
end

labels   = reshape(labels, [PRowNum, PColNum]);
end

function cleanZP = updatePatchbyGauss(noisyPs, GMM, PLabels, noiseCovs, condProb)
currPs   = noisyPs(:,:,:,condProb(1));
noiseCov = noiseCovs(:,:,:,condProb(1));
[ZPRowNum, ZPColNum, PSize] = size(currPs);
ZPNum    = ZPRowNum*ZPColNum;
currPs   = reshape(currPs, [ZPNum, PSize])';
PLabels  = reshape(PLabels, [ZPNum, 1])';

cleanZP  = 0;
existLabels = unique(PLabels);
if PSize == size(GMM.means, 1)
    covs     = GMM.covs;
    mus      = GMM.means;
    
    for Ci = existLabels
        Cinds   = PLabels == Ci;
        Cov     = covs(:,:,Ci);
        Mu      = mus(:,Ci);
        invCov  = inv(Cov + noiseCov);
        cleanZP = cleanZP + invCov * (Cov*currPs + noiseCov*Mu) .* Cinds;
    end
else
    ZP2   = noisyPs(:,:,:,condProb(2:end));
    [~, ~, ~, condNum] = size(ZP2);
    ZP2   = reshape(ZP2, [ZPNum, PSize*condNum])';
    
    nCov2 = [];
    for fn = 2:condNum+1
        nCov2 = blkdiag(nCov2, noiseCovs(:,:,:,condProb(fn)));
    end
    
    cov11 = GMM.covs(1:PSize, 1:PSize, :);
    cov21 = GMM.covs(PSize+1:end, 1:PSize, :);
    cov12 = GMM.covs(1:PSize, PSize+1:end, :);
    cov22 = GMM.covs(PSize+1:end, PSize+1:end, :) + nCov2;
    
    mu1   = GMM.means(1:PSize, :);
    mu2   = GMM.means(PSize+1:end, :);
    
    for Ci = existLabels
        Cinds   = PLabels == Ci;
        CovTmp  = cov12(:,:,Ci)/cov22(:,:,Ci);
        Mu      = CovTmp*(ZP2 - mu2(:,Ci)) + mu1(:,Ci);
        Cov     = cov11(:,:,Ci) - CovTmp*cov21(:,:,Ci);
        invCov  = inv(Cov + noiseCov);
        cleanZP = cleanZP + invCov * (Cov*currPs + noiseCov*Mu).* Cinds;
    end
end

cleanZP  = reshape(cleanZP', [ZPRowNum, ZPColNum, PSize]);
end

function condGMM = prepareCondProb(jointGMM, jointProb, condProb)
[PSize,k] = size(jointGMM.means);
PSize     = PSize ./ numel(jointProb);

bounds    = PSize * ones(1,numel(jointProb));
muSplit   = mat2cell(jointGMM.means, bounds, k);
covSplit  = mat2cell(jointGMM.covs, bounds, bounds, k);

[~,transfer]=ismember(condProb, jointProb);

covNew   = cell(numel(condProb), numel(condProb));
for col = 1:numel(condProb)
    for row = 1:numel(condProb)
        if row == col
            covNew{row,col} = covSplit{transfer(row), transfer(col)};
        elseif row < col
            if transfer(row) < transfer(col)
                covNew{row,col} = covSplit{transfer(row), transfer(col)};
            else
                covNew{row,col} = permute(covSplit{transfer(col), transfer(row)}, [2,1,3]);
            end
        else
            if transfer(row) > transfer(col)
                covNew{row,col} = covSplit{transfer(row), transfer(col)};
            else
                covNew{row,col} = permute(covSplit{transfer(col), transfer(row)}, [2,1,3]);
            end
        end
    end
end

condGMM = struct('mixweights', jointGMM.mixweights, ...
    'means',      cell2mat(muSplit(transfer, :)), ...
    'covs',       cell2mat(covNew));
end