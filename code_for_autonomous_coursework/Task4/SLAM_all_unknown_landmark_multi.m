function [] = EKF_localization_experiment(landmark_count, num_runs)
% EKF_localization_experiment
% Investigate the impact of different numbers of landmarks on SLAM robot localization
% Input parameters:
%   landmark_count : Specify the number of landmarks (e.g., 20)
%   num_runs       : Number of runs (e.g., 1000), default is 1000 runs
%
% Each run outputs a single RMSE, and finally the average RMSE over all runs

if nargin < 1
    landmark_count = 20; % Default number of landmarks
end
if nargin < 2
    num_runs = 1000;     % Default run count of 1000 times
end

all_rmse = zeros(num_runs,1);
fprintf('Starting experiment: landmark count = %d, number of runs = %d\n', landmark_count, num_runs);

for run_idx = 1:num_runs
    rmse = EKF_localization_once(landmark_count);
    all_rmse(run_idx) = rmse;
    fprintf('Run number %d: RMSE = %f\n', run_idx, rmse);
end

avg_rmse = mean(all_rmse);
fprintf('Average RMSE over %d runs = %f\n', num_runs, avg_rmse);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rmse = EKF_localization_once(No_of_Landmarks)
% EKF_localization_once
% Single run EKF SLAM simulation (all plotting code removed to ensure faster execution)
%
% Input:
%   No_of_Landmarks : Specified number of landmarks
% Output:
%   rmse : RMSE of the robot's position during the simulation

tic;
time = 0;
endTime = 50; % Simulation duration (seconds)

global dt;
global localizer;
global Qsigma;
global Rsigma;

% Initial state settings
xEst = [0; 0; 0];   % Estimated state [x; y; theta]
xGnd = xEst;       % Ground truth state
xOdom = xGnd;      % Odometry state (without measurement update)
P = zeros(3,3);    % Initial covariance matrix
dt = 0.1;          % Sampling time (seconds)

%% Noise and sensor parameters
Qsigma = diag([0.1, 0.01]).^2;  % Control noise covariance
Rsigma = diag([0.01, 0.01]).^2; % Observation noise covariance
MAX_RANGE = 1;                 % Maximum sensor range

%% Landmark position generation
worldsize = 1.5;
landMarks = [];
for i = 1:No_of_Landmarks
    landMarks = [landMarks; -worldsize + 2*worldsize*rand, -0.5 + 2*worldsize*rand];
end
% Save landmarks (optional)
%save('lastlandmarks.mat', 'landMarks'); 

%% Variable initialization
errs = [];
setup_localizer();
nSteps = ceil((endTime - time)/dt);

%% Noise tuning parameters
Qtune = Qsigma;
Rtune = Rsigma;

%% Data association parameters
gating_threshold = 15;  % Mahalanobis threshold
method = 2;             % 2: Mahalanobis distance (alternatively 1: Euclidean distance)

%% Outlier rejection parameter
max_update = 1e1; % Threshold for measurement update change

%% Main loop
for i = 1:nSteps
    time = time + dt;
    % Obtain control input (with exponential rise)
    u = doControl(time);
    % Simulate real control input with added noise
    ureal = u + sqrt(Qsigma)*randn(2,1);
    
    % Update ground truth state (nonlinear motion model)
    xGnd = xGnd + [ dt*cos(xGnd(3))*ureal(1); dt*sin(xGnd(3))*ureal(1); dt*ureal(2)];
                
    % Generate observations (range and bearing); measurements are generated only for landmarks within sensor range
    z = [];
    if No_of_Landmarks > 0
        for iz = 1:size(landMarks,1)
            d = norm(xGnd(1:2)' - landMarks(iz,:));
            alpha = atan2(landMarks(iz,2) - xGnd(2), landMarks(iz,1) - xGnd(1)) - xGnd(3);
            if d < MAX_RANGE
                z = [z; [ max(0, d + sqrt(Rsigma(1,1))*randn), alpha + sqrt(Rsigma(2,2))*randn, landMarks(iz,:)]];
            end
        end
    end

    % Pure odometry state update
    xOdom = xOdom + [ dt*cos(xOdom(3))*u(1); dt*sin(xOdom(3))*u(1); dt*u(2) ]; 
                  
    %% Prediction update
    n_states = length(xEst);
    A_rb = [ 1, 0, -u(1)*sin(xEst(3));
             0, 1,  u(1)*cos(xEst(3));
             0, 0,  1 ];
    B_rb = [ cos(xEst(3)), 0;
             sin(xEst(3)), 0;
             0,            1 ];
    A = eye(n_states);
    A(1:3,1:3) = A_rb;
    B = zeros(n_states,2);
    B(1:3,:) = B_rb;
    P = A * P * A' + B * Qtune * B';
    xEst(1:3) = xEst(1:3) + [dt*cos(xEst(3))*u(1); dt*sin(xEst(3))*u(1); dt*u(2)];

    %% Measurement update
    if ~isempty(z)
        for iz = 1:size(z,1)
            d_meas = z(iz,1);
            bearing_meas = z(iz,2);
            
            % Data association
            i_lm = dataAssociation(xEst, P, d_meas, bearing_meas, Rtune, gating_threshold, method);
            
            if i_lm == 0  % New landmark
                n_states = length(xEst);
                lm_x = xEst(1) + d_meas*cos(xEst(3)+bearing_meas);
                lm_y = xEst(2) + d_meas*sin(xEst(3)+bearing_meas);
                xEst = [xEst; lm_x; lm_y];
                
                % Compute Jacobian matrices for state augmentation
                G_v = [ 1, 0, -d_meas*sin(xEst(3) + bearing_meas);
                        0, 1,  d_meas*cos(xEst(3) + bearing_meas) ];
                G_z = [ cos(xEst(3) + bearing_meas), -d_meas*sin(xEst(3) + bearing_meas);
                        sin(xEst(3) + bearing_meas),  d_meas*cos(xEst(3) + bearing_meas) ];
                    
                P_xx = P(1:3, 1:3);
                P_xm = P(1:3, 4:n_states);
                    
                P_mm_1 = G_v*P_xx*G_v' + G_z*Rtune*G_z';
                P_mm_2 = G_v*P_xm;
                P_mm_3 = P_mm_2';
                P_xm_1 = P_xx * G_v';
                P_mx_1 = P_xm_1';
                    
                P_last = P;
                P = zeros(n_states+2, n_states+2);
                P(1:n_states, 1:n_states) = P_last;
                P(1:3, n_states+1:n_states+2) = P_xm_1;
                P(n_states+1:n_states+2, 1:3) = P_mx_1;
                if n_states > 3
                    P(4:n_states, n_states+1:n_states+2) = P_mm_2';
                    P(n_states+1:n_states+2, 4:n_states) = P_mm_3';
                end
                P(n_states+1:n_states+2, n_states+1:n_states+2) = P_mm_1;
                
            else  % Existing landmark
                n_states = length(xEst);
                id_x = 3 + 2*(i_lm-1) + 1;
                id_y = id_x + 1;
                    
                dx = xEst(id_x) - xEst(1);
                dy = xEst(id_y) - xEst(2);
                r_pred = sqrt(dx^2 + dy^2);
                bearing_pred = wrapToPi(atan2(dy, dx) - xEst(3));
                    
                y_innov = [d_meas - r_pred;
                           wrapToPi(bearing_meas - bearing_pred)];
                    
                H_rb = [ -dx/r_pred, -dy/r_pred, 0;
                         dy/(r_pred^2), -dx/(r_pred^2), -1];
                H_l = [ dx/r_pred, dy/r_pred;
                       -dy/(r_pred^2), dx/(r_pred^2)];
                    
                H = zeros(2, n_states);
                H(:,1:3) = H_rb;
                H(:,id_x:id_y) = H_l;
                    
                S = H * P * H' + Rtune;
                K = P * H' / S;
                    
                if norm(K * y_innov) < max_update
                    xEst = xEst + K * y_innov;
                    P = (eye(n_states) - K * H) * P;
                end
            end
        end
    end
    
    % Record current position error
    errs = [errs, norm(xGnd(1:2)' - xEst(1:2)')];
    
    % To speed up execution, all plotting and data storage code has been removed
end

% Compute RMSE: Root Mean Square Error
rmse = sqrt(mean(errs.^2));
toc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following are helper functions, no plotting code involved

function radian = toRadian(degree)
    radian = degree/180*pi;
end

function [ u ] = doControl( time )
    % Control input rises exponentially over time to steady state
    T = 10; % Time constant [seconds]
    V = 0.1; % Linear speed [m/s]
    yawrate = 5; % Angular speed [deg/s]
    u = [ V*(1-exp(-time/T)), toRadian(yawrate)*(1-exp(-time/T)) ]';
end

function idx_landmark = dataAssociation(xEst, P, r_meas, bearing_meas, R, gating_threshold, method)
% Data association function: determine whether the current measurement belongs to an existing landmark or a new landmark
if nargin < 7
    method = 2;
end

n_state = length(xEst);
n_landmarks = (n_state - 3) / 2;
if n_landmarks == 0
    idx_landmark = 0;
    return;
end

rx = xEst(1);
ry = xEst(2);
rtheta = xEst(3);
z_actual = [r_meas; bearing_meas];

dists = zeros(n_landmarks, 1);
for i = 1:n_landmarks
    idxX = 3 + 2*(i-1) + 1;
    idxY = idxX + 1;
    lm_x = xEst(idxX);
    lm_y = xEst(idxY);
        
    dx = lm_x - rx;
    dy = lm_y - ry;
    pred_range = sqrt(dx^2 + dy^2);
    pred_bearing = wrapToPi(atan2(dy, dx) - rtheta);
        
    z_pred = [pred_range; pred_bearing];
    innov = z_actual - z_pred;
    innov(2) = wrapToPi(innov(2));
        
    switch method
        case 1
            dists(i) = norm(innov);
        case 2
            H_r = [ -dx/pred_range, -dy/pred_range, 0;
                     dy/(pred_range^2), -dx/(pred_range^2), -1];
            H_l = [ dx/pred_range, dy/pred_range;
                   -dy/(pred_range^2), dx/(pred_range^2)];
            H = zeros(2, n_state);
            H(:,1:3) = H_r;
            H(:, idxX:idxY) = H_l;
            S = H * P * H' + R;
            dists(i) = innov' / S * innov;
        otherwise
            error('Unknown method parameter, please choose 1 or 2');
    end
end

[dist_min, i_min] = min(dists);
if dist_min > gating_threshold
    idx_landmark = 0;
else
    idx_landmark = i_min;
end
end

function setup_localizer()
    global localizer;
    localizer.time = [];
    localizer.xEst = [];
    localizer.xGnd = [];
    localizer.xOdom = [];
    localizer.z = [];
    localizer.PEst = [];
    localizer.u = [];
    localizer.landmarkEst = [];
    localizer.landmarkTrue = [];
end
