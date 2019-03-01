function Raw2Tif(RawFilesdir,TifFilesdir)
%Reads the .raw file from RawFilesdir and  writest to .tif files in
%TifFilesdir

files = dir(RawFilesdir) ;    % you are in folder of raw files
N = length(files) ;   % total number of files 
% loop for each file 
for i = 3:N
    RawImdir=join([RawFilesdir,'/',files(i).name]);
    %RawImdi= [Rawdir,'/',files(i).name]
    Im=rawimread(RawImdir);
    Im=Im./max(Im(:));
    tifext = replace(files(i).name,'.raw','.tif'); %To get .tif extension
    TifImdir=join([TifFilesdir,'/',tifext]);
    imwrite(Im,TifImdir);
    size(tifext);
    
end
end
