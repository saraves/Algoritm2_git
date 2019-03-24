function binaryImage = Tif2Bin(filename, name)
    % Input: tif image
    % Returns binary image

    t = Tiff(filename, 'r');                        % Read tif file
    imageData = read(t);
    imageData = imageData(:,:,1:1);                 % Change to correct dimensions
    imageData = im2double(imageData); 
    imageData = imageData./max(imageData(:));       % Normalize
    
    
    
    realIm = rawimread(strcat('RawFilesdir/', name, '.raw'));       % Real image
    realIm = realIm./max(realIm(:)); 
    mask = imageData-realIm;                                  % Extract mask only
    
    binaryImage = imbinarize(mask);                           % Convert to binary

end                                

    

