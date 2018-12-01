% display multi-slice
% usage:
%       viewslice(image,matrix,rotate,flip,range,sliceRange,oFormat,txLabel)
%       range = [min,max] of colormap
%       oFormat = [X,Y] display X(Y) slices in X(Y)-axis
%
% KH Chuang @ LFMI/NIH
% 11/12/2003 ver 1.0
% 4/23/2005  ver 2.0: don't need montage(); add oFormat
% 12/20/2005 ver 2.1: add txLabel and allow empty options

function viewslice(itmp,mtx,rot,flip,range,sliceRange,oFormat,txLabel);

if ndims(itmp)~=3
    itmp=reshape(itmp,mtx(1),mtx(2),mtx(3));
end
imin=0;
imax=max(max(max(max(itmp))));
if exist('range','var')~=0
    if isempty(range)==0
        imin=range(1); imax=range(2);
    end
end
if exist('rot')==0 | isempty(rot)~=0;rot=0;end
if exist('flip')==0| isempty(flip)~=0;flip=0;end
if exist('sliceRange')==0| isempty(sliceRange)~=0;sliceRange=1:mtx(3);end
if(rot==1)
    itmp=permute(itmp,[2,1,3]);
    mtx=size(itmp);
end
if(flip==1)
    itmp=flipdim(itmp,1);
end

nsl=length(sliceRange);
if exist('oFormat')~=0 & isempty(oFormat)==0
    xsize=oFormat(1);
    ysize=oFormat(2);
    if xsize*ysize < nsl
        ysize=ceil(nsl/xsize);
    end
else
    xsize=round(sqrt(nsl));
    ysize=ceil(nsl/xsize);
end

if exist('txLabel')==0| isempty(txLabel)~=0
    txLabel=0;
end

ystart=1;yend=xsize;
otmp=reshape(itmp(:,:,sliceRange(ystart:yend)),mtx(1),[]);
for n=2:ysize
    ystart=ystart+xsize;
    yend=yend+xsize;
    if yend>nsl
        otmp=[otmp;reshape(itmp(:,:,sliceRange(ystart:nsl)),mtx(1),[]),zeros(mtx(1),mtx(2)*(yend-nsl))];
    else
        otmp=[otmp;reshape(itmp(:,:,sliceRange(ystart:yend)),mtx(1),[])];
    end
end
h=imagesc(otmp,[imin,imax]);axis off;axis image;hb=colorbar;

if txLabel==1
    coldist=mtx(1); rowdist=mtx(2);
    col=get(h,'XData'); col=col(2)/coldist;
    for n=0:nsl-1
        text(rem(n,col)*coldist+2,floor(n/col)*rowdist+10,int2str(sliceRange(n+1)),'FontWeight','bold','Color','y');
    end
end
