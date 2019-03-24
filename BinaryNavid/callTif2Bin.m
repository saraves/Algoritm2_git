clc; clear; close all;

% Testar att iterera genom mappar med filer

dirName = 'MaskedNavid';

tifFiles = dir([char(dirName), '/*.tif']);    % tif-files in folder 1
numFiles = length(tifFiles);                         % Number of tif-files

for i = 1:numFiles

    % Iterate through all the files and binarize them
    [filepath, name, ext] = fileparts(tifFiles(i).name);    % extract filenam
    filename = strcat(dirName, '/', name, ext);
    image = Tif2Bin(filename, name);
    
    % Save finalImage to FinalData.dir
    imagedir = join(['FinalData/',tifFiles(i).name]);
    imwrite(image, imagedir);
end

