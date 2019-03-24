clc; clear; close all;

% Testar att iterera genom mappar med filer

folder = 'Person';
N = 3;                                      % Copies of same image
directories = strings(N,1);                 % Create string array

for n = 1:N
    dirName = sprintf('%s%d', folder, n);   % Save all subfolders
    directories(n,1) = dirName;
end

tifFiles = dir([char(directories(1)), '/*.tif']);    % tif-files in folder 1
numFiles = length(tifFiles);                         % Number of tif-files



for i = 1:numFiles
    
    currentImage = 0;

    for j = 1:N
        % Iterate through all the files & sum all the image versions
        [filepath, name, ext] = fileparts(tifFiles(i).name);    % extract filename
        currentFilename = strcat(directories(j,1), '/', name, ext);
        currentImage = currentImage + Tif2Bin(currentFilename, name);
        
        
    end
   
    finalImage = compare(currentImage);    % Combined image to binary
    
    % Save finalImage to FinalData.dir
    finalImagedir = join(['FinalData/',tifFiles(i).name]);
    imwrite(finalImage, finalImagedir);
end

