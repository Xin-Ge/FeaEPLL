function [y, counts] = patches2im(patches, patchSize, stride, dilate, overlapMethod, counts)
if nargin < 6; counts = []; end
if nargin < 5; overlapMethod = 'average'; end
if nargin < 4; dilate = 1;  end
if nargin < 3; stride = 1;  end
if nargin < 2; error('Do not define the size of patch'); end

if length(patchSize) == 1; patchSize = [patchSize, patchSize]; end
if length(stride)    == 1; stride    = [stride,  stride];      end
if length(dilate)    == 1; dilate    = [dilate,  dilate];      end

ksize2  = patchSize(1) * patchSize(2);
%%
if ~isempty(patches)
    [xrawNum, xcolNum, xchaNum, xbatNum]  = size(patches);
else
    error('not enough inputs');
end
ker_rinds  = 1 : dilate(1) : (patchSize(1)-1)*dilate(1)+1;
ker_cinds  = 1 : dilate(2) : (patchSize(2)-1)*dilate(2)+1;
imb_rinds  = 0 : stride(1) : (xrawNum-1)*stride(1);
imb_cinds  = 0 : stride(2) : (xcolNum-1)*stride(2);
imbed_rnum = ker_rinds(end) + imb_rinds(end);
imbed_cnum = ker_cinds(end) + imb_cinds(end);
ker_rinds  = repmat(ker_rinds, [1, patchSize(2)]);
ker_cinds  = repmat(ker_cinds, [patchSize(1), 1]);
ker_rinds  = ker_rinds(:);
ker_cinds  = ker_cinds(:);

%%
if isempty(counts)
    counts = zeros([imbed_rnum, imbed_cnum], class(gather(patches)));
    bgzero = counts;
    S.type = '()';
    S.subs = {':', ':'};
    for nn = 1 : ksize2
        S.subs{1} = ker_rinds(nn) + imb_rinds;
        S.subs{2} = ker_cinds(nn) + imb_cinds;
        counts  = counts + subsasgn(bgzero, S, 1);
    end
end
S2.type = '()';
S2.subs = {':', ':', ':', ':'};
S1.type = '()';
S1.subs = {':', ':', ':', ':'};

if isa(patches, 'gpuArray')
    y = gpuArray.zeros([imbed_rnum, imbed_cnum, xchaNum/ksize2, xbatNum], classUnderlying(patches));
else
    y = zeros([imbed_rnum, imbed_cnum, xchaNum/ksize2, xbatNum], class(patches));
end
bgzero = y;
for nn = 1 : ksize2
    S1.subs{1} = ker_rinds(nn) + imb_rinds;
    S1.subs{2} = ker_cinds(nn) + imb_cinds;
    S2.subs{3} = nn : ksize2 : xchaNum;
    y  = y + subsasgn(bgzero, S1, subsref(patches, S2));
end
if strcmp(overlapMethod, 'average')
    y = bsxfun(@rdivide, y, counts);
end
