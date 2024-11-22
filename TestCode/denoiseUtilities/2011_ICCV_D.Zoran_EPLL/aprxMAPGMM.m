function [Xhat] = aprxMAPGMM(Y,patchSize,noiseSD,imsize,GS,excludeList,SigmaNoise,Pyorig,Pat)
% approximate GMM MAP estimation - a single iteration of the "hard version"
% EM MAP procedure (see paper for a reference)
%
% Inputs:
%   Y - the noisy patches (in columns)
%   noiseSD - noise standard deviation
%   imsize - size of the original image (not used in this case, but may be
%   used for non local priors)
%   GS - the gaussian mixture model structure
%   excludeList - used only for inpainting, misleading name - it's a list
%   of patch indices to use for estimation, the rest are just ignored
%   SigmaNoise - if the noise is non-white, this is the noise covariance
%   matrix
%
% Outputs:
%   Xhat - the restore patches


% handle exclusion list - used for inpainting
if ~exist('excludeList','var')
    excludeList = [];
end

% Supports general noise covariance matrices
if ~exist('SigmaNoise','var') || isempty(SigmaNoise)
    SigmaNoise = noiseSD^2*eye(patchSize^2);
end

if ~exist('Pat','var') || isempty(Pat)
    Pat = 1;
else
    if Pat~=1 && (~exist('Pyorig','var') || isempty(Pyorig))
        error('Not Enough Inputs')
    end
end

if ~exist('Pyorig','var') || isempty(Pyorig)
    Pyorig = Y;
end

if ~isempty(excludeList)
    T = Y;
    Y = Y(:,excludeList);
end

% remove DC component
switch Pat
    case 2
        meanYClass = mean(Pyorig);
        meanYClear = mean(Pyorig);
    case 3
        meanYClass = mean(Pyorig);
        meanYClear = mean(Y);
    case 4
        meanYClass = mean(Y);
        meanYClear = mean(Pyorig);
    otherwise
        meanYClass = mean(Y);
        meanYClear = mean(Y);
end
YClass = bsxfun(@minus,Y,meanYClass);
YClear = bsxfun(@minus,Y,meanYClear);

% calculate assignment probabilities for each mixture component for all
% patches
GS2 = GS;
PYZ = zeros(GS.nmodels,size(YClass,2));
for i=1:GS.nmodels
    GS2.covs(:,:,i) = GS.covs(:,:,i) + SigmaNoise;
    PYZ(i,:) = log(GS.mixweights(i)) + loggausspdf2(YClass,GS2.covs(:,:,i));
end

% find the most likely component for each patch
[~,ks] = max(PYZ);

% and now perform weiner filtering
Xhat = zeros(size(YClear));
for i=1:GS.nmodels
    inds = find(ks==i);
    Xhat(:,inds) = ((GS.covs(:,:,i)+SigmaNoise)\(GS.covs(:,:,i)*YClear(:,inds) + SigmaNoise*repmat(GS.means(:,i),1,length(inds))));
end

% handle exclusion list stuff (inpainting only)
if ~isempty(excludeList)
    tt = T;
    tt(:,excludeList) = bsxfun(@plus,Xhat,meanYClear);
    Xhat = tt;
else
    Xhat = bsxfun(@plus,Xhat,meanYClear);
end
    
    
