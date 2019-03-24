clc; clear; close all;

% Testar att iterera genom mappar med filer

dirName = ('MaskedNavid');

tifFiles = dir([char(dirName), '/*.tif']);    % tif-files in folder 1
numFiles = length(tifFiles);                         % Number of tif-files



for i = 1:numFiles

    % Iterate through all the files & sum all the image versions
    [filepath, name, ext] = fileparts(tifFiles(i).name);    % extract filename
    filename = strcat(dirName, '/', name, ext);
    image = Tif2Bin(filename, name);
   
    finalImage = compare(currentImage);    % Combined image to binary
    
    % Save finalImage to FinalData.dir
    finalImagedir = join(['FinalData/',tifFiles(i).name]);
    imwrite(finalImage, finalImagedir);
end

