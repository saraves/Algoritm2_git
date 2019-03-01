clear all; close; clc;

t = Tiff('TifFiles/B12.tif','r');
imageData = read(t);
%imageData = imageData(:,:,1:3);
imageData=im2double(imageData);
imageData=imageData./max(imageData(:));

size(imageData)


Im = rawimread('RawFilesdir/E3.raw');
Im=Im./max(Im(:));
k=imageData-Im;

figure()
ax1=subplot(2,2,1)
imagesc(imageData)
axis image;


ax2=subplot(2,2,2)
imagesc(Im)
colormap(gray)
axis image;

ax3=subplot(2,2,3)
imagesc(k)
axis image;

binary = imbinarize(k, 0.001);
ax4=subplot(224)
imagesc(binary)
axis image;

linkaxes([ax1,ax2,ax3, ax4],'xy');


