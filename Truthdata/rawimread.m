function I = rawimread( fname, endi, trans )
if nargin < 3
    % Default to showing the image without the mirroring  effect. Optional
    % because some of the ground truth data was generated with mirroring
    % activated.
    trans = false;
    if nargin < 2
        % Default to Big Endian
        endi = 'l';
    end
end
    % Find out length of sides (only works for square, uint16, purely RAW
    % images)
    d = dir( fname );
    sideSize = sqrt( d.bytes / 2 ); % /2, num pix, sqrt, num pix side
    % Open file to read
    fid = fopen( fname, 'r' );
    % Read data, assume 2048 size for now
    I = fread( fid, sideSize*ones(1,2), 'uint16', 0, endi );
    % Close file
    fclose( fid );
    % Transpose image (it will then not be mirrored with respect to
    % reality, at least with Allied Vision and co.)
    if trans
        I = I';
    end
end