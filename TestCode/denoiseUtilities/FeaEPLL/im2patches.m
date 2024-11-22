function [patches] = im2patches(im, patchSize, stride, dilate)
if nargin < 2; error('Do not define the size of patch'); end
if nargin < 3; stride = 1; end
if nargin < 4; dilate = 1; end

if length(patchSize) == 1; patchSize = [patchSize, patchSize]; end
if length(stride)  == 1;   stride    = [stride,  stride];      end
if length(dilate)  == 1;   dilate    = [dilate,  dilate];      end

psize2  = patchSize(1) * patchSize(2);
%%
if ~isempty(im)
    [xrawNum, xcolNum, xchaNum, xbatNum]  = size(im);
else
    error('not enough inputs');
end
ker_rinds  = 0 : dilate(1) : (patchSize(1)-1)*dilate(1);
ker_cinds  = 0 : dilate(2) : (patchSize(2)-1)*dilate(2);
imb_rinds  = 1 : stride(1) : xrawNum-ker_rinds(end);
imb_cinds  = 1 : stride(2) : xcolNum-ker_cinds(end);
imbed_rnum = numel(imb_rinds);
imbed_cnum = numel(imb_cinds);
ker_rinds  = repmat(ker_rinds, [1, patchSize(2)]);
ker_cinds  = repmat(ker_cinds, [patchSize(1), 1]);
ker_rinds  = ker_rinds(:);
ker_cinds  = ker_cinds(:);

%%
S2.type = '()';
S2.subs = {':', ':', ':', ':'};
S1.type = '()';
S1.subs = {':', ':', ':', ':'};

if isa(im, 'gpuArray')
    patches = gpuArray.zeros([imbed_rnum, imbed_cnum, xchaNum*psize2, xbatNum], classUnderlying(im));
else
    patches = zeros([imbed_rnum, imbed_cnum, xchaNum*psize2, xbatNum], class(im));
end
for nn = 1 : psize2
    S1.subs{1} = ker_rinds(nn) + imb_rinds;
    S1.subs{2} = ker_cinds(nn) + imb_cinds;
    S2.subs{3} = nn : psize2 : xchaNum*psize2;
    patches  = subsasgn(patches, S2, subsref(im, S1));
end
