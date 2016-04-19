% Matlab demo code accompanying the CVPR11 paper [C. Rhemann, A. Hosni,
% M. Bleyer, C. Rother, M. Gelautz, Fast Cost-Volume Filtering for Visual
% Correspondence and Beyond, CVPR11]
% Contributed by Christoph Rhemann (rhemann@ims.tuwien.ac.at)
%
% Please note that the running time and results reported in the paper is from a CUDA
% implementation. Hence, results and running times do not match this Matlab
% implementation.

function output = runStereoMatcher(left,right,numDisp)

UseParallelToolbox = false; % Set true if you want to take advantage of the Matlab parallel computing toolbox
ParallelWorkers = 2; % How many workers should be used by the parallel computing toolbox (should be equal or less the number of available CPU cores)

% Set up parallel computing toolbox
if (UseParallelToolbox)
    isOpen = matlabpool('size') > 0;
    if (isOpen)
        matlabpool close force local
    end
    matlabpool(ParallelWorkers)
end

% Parameter settings
r = 19;                  % filter kernel in eq. (3) has size r \times r
eps = 0.0001;           % \epsilon in eq. (3)
thresColor = 0.0028;     % \tau_1 in eq. (5)
thresGrad = 0.008;      % \tau_2 in eq. (5)
gamma = 0.1;           % (1- \alpha) in eq. (5)
threshBorder = 3/255;   % some threshold for border pixels
gamma_c = 0.1;          % \sigma_c in eq. (6)
gamma_d = 9;            % \sigma_s in eq. (6)
r_median = 19;          % filter kernel of weighted median in eq. (6) has size r_median \times r_median

Ir = double(imread(right))/255;
Il = double(imread(left))/255;
% Compute disparity map for middlebury test images (vision.middlebury.edu/stereo/)
output = example_referenceForCVPR11(Il,Ir,numDisp,r,eps,thresColor,thresGrad,gamma,threshBorder,gamma_c,gamma_d,r_median,UseParallelToolbox);
end
