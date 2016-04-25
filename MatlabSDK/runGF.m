clear variables;
warning('off','all')
close all;

% Are you going to use the training or test set?
imgset = 'training';
%imgset = 'test';

% Specify which resolution you are using for the stereo image set (F, H, or Q?)
imgsize = 'Q';
%imgsize = 'H';
%imgsize = 'F';

% What are you calling your method?
methodname = 'GF';

mkdir('MiddEval3results');
mkdir(['MiddEval3results/',imgset,imgsize]);
if strcmp(imgset,'training')
    image_names{1} = 'Adirondack';
    image_names{2} = 'ArtL';
    image_names{3} = 'Jadeplant';
    image_names{4} = 'Motorcycle';
    image_names{5} = 'MotorcycleE';
    image_names{6} = 'Piano';
    image_names{7} = 'PianoL';
    image_names{8} = 'Pipes';
    image_names{9} = 'Playroom';
    image_names{10} = 'Playtable';
    image_names{11} = 'PlaytableP';
    image_names{12} = 'Recycle';
    image_names{13} = 'Shelves';
    image_names{14} = 'Teddy';
    image_names{15} = 'Vintage';
    ndisp = [290, 256, 640, 280, 280, 260, 260, 300, 330, 290, 290, 260, 240, 256, 760];
else
    image_names{1} = 'Australia';
    image_names{2} = 'AustraliaP';
    image_names{3} = 'Bicycle2';
    image_names{4} = 'Classroom2';
    image_names{5} = 'Classroom2E';
    image_names{6} = 'Computer';
    image_names{7} = 'Crusade';
    image_names{8} = 'CrusadeP';
    image_names{9} = 'Djembe';
    image_names{10} = 'DjembeL';
    image_names{11} = 'Hoops';
    image_names{12} = 'Livingroom';
    image_names{13} = 'Newkuba';
    image_names{14} = 'Plants';
    image_names{15} = 'Staircase';
    ndisp = [290, 290, 250, 610, 610, 256, 800, 800, 320, 320, 410, 320, 570, 320, 450];
end

ErrorRate = zeros(1,15);
for im_num = 1:15
    I{1} = imread(['~/MiddEval3/',imgset,imgsize,'/',image_names{im_num},'/im0.png']);
    I{2} = imread(['~/MiddEval3/',imgset,imgsize,'/',image_names{im_num},'/im1.png']);
    
    I{1} = double(I{1})/255;
    I{2} = double(I{2})/255;
    
    % Adjust the range of disparities to the chosen resolution
    if imgsize == 'Q'
        DisparityRange = round(ndisp(im_num)/4);
    elseif imgsize == 'H'
        DisparityRange = round(ndisp(im_num)/2);
    else
        DisparityRange = round(ndisp(im_num));
    end
    
    tic;
    %--------------- Insert your stereo matching routine here ------------%
    r = 19;                  % filter kernel in eq. (3) has size r \times r
    eps = 0.0001;           % \epsilon in eq. (3)
    thresColor = 0.0028;     % \tau_1 in eq. (5)
    thresGrad = 0.008;      % \tau_2 in eq. (5)
    gamma = 0.1;           % (1- \alpha) in eq. (5)
    threshBorder = 3/255;   % some threshold for border pixels
    gamma_c = 0.1;          % \sigma_c in eq. (6)
    gamma_d = 9;            % \sigma_s in eq. (6)
    r_median = 19;          % filter kernel of weighted median in eq. (6) has size r_median \times r_median
    
    % Compute disparity map for middlebury test images (vision.middlebury.edu/stereo/)
    DisparityMap = example_referenceForCVPR11(I{1},I{2},DisparityRange,r,eps,thresColor,thresGrad,gamma,threshBorder,gamma_c,gamma_d,r_median,false);
    %---------------------------------------------------------------------%
    time_taken = toc;
    imshow([DisparityMap/DisparityRange]);
    drawnow;
    % This assumes that inconsistent disparities are given a value <= 0
    
    % If possible, compute the error rate
    if strcmp(imgset,'training')
        GT = readpfm(['~/MiddEval3/training',imgsize,'/',image_names{im_num},'/disp0GT.pfm']);
        mask = imread(['~/MiddEval3/training',imgsize,'/',image_names{im_num},'/mask0nocc.png']);
        mask = mask == 255;
        Error = abs(DisparityMap - GT) > 1;
        Error(~mask) = 0;
        ErrorRate(im_num) = sum(Error(:))/sum(mask(:));
        fprintf('%s = %f\n', image_names{im_num}, ErrorRate(im_num));
    end
    
    % Output disparity maps and timing results for Middlebury evaluation
    pfmwrite(single(DisparityMap), ['~/MiddEval3/',imgset,imgsize,'/',image_names{im_num},'/disp0',methodname,'.pfm']);
    fid = fopen(['~/MiddEval3/',imgset,imgsize,'/',image_names{im_num},'/time',methodname,'.txt'],'w');
    
    fprintf(fid,'%f',time_taken);
    fclose(fid);
end

if strcmp(imgset,'training')
    ErrorRateMean = (sum(ErrorRate([1,2,3,4,5,6,8,11,12,14])) + 0.5*sum(ErrorRate([7,9,10,13,15])))/12.5;
end

fprintf('Overall = %f\n', ErrorRateMean);

