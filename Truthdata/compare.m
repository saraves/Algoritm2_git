function total = compare(image)
    % Input combined images
    % Returns the combined binary image
    
    [m, n] = size(image);
    
    % iterate through columns m and rows n (col, row) = (m, n) = (i, j)
    for i = 1:m
        for j = 1:n
            if image(i, j) == 1
                % Replace with zero
                image(i, j) = 0;
            elseif image(i, j) > 1
                % Replace with one
                image(i, j) = 1;
            end
        end
    end
    
    total = image;

end
