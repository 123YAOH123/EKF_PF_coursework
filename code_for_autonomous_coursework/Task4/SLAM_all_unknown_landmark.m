function [] = EKF_localization()
% EKF-SLAM for all unknown landmark
% Modified from the original EKF code(originally produced by Zhixin Zhang, Josh Bettles, and JC.).
% Version: 15 Mar 2025 

%% initialization
close all;
clear all;

disp('EKF SLAM program start!!')

tic;
time = 0;
endTime = 50; % seconds

global dt;
global localizer;
global Qsigma;
global Rsigma;

% Initial robot state (3x1): [x; y; theta]
xEst = [0; 0; 0];   % Estimated State [x y yaw]'
xGnd = xEst;       % GroundTruth State (the yaw angle is not 
% restricted to [-pi,pi])
xOdom = xGnd;      % Odometry-only = Dead Reckoning 
% (model with no measurements)
P = zeros(3,3);    % initial covariance (3x3)
dt = 0.1; % second

%% real noises and sensor range 
Qsigma = diag([0.1, 0.01]).^2;  % process (control) noise
Rsigma = diag([0.01, 0.01]).^2; % observation noise
MAX_RANGE = 1; % maximum sensor range (Large value for always measurable)

%% landmark positions 
No_of_Landmarks = 20;       
worldsize = 1.5; 
landMarks = [];
for i = 1:No_of_Landmarks
    landMarks = [landMarks; -worldsize+2*worldsize*rand, -0.5+2*worldsize*rand];
end
% Last marks are saved in case you want to reuse them:
%save('lastlandmarks.mat', 'landMarks'); 

%% initialisation of variables:
errs = [];
setup_localizer();
nSteps = ceil((endTime - time)/dt);

%% Suggested noises 
Qtune = Qsigma;  % default tuning option
Rtune = Rsigma;  % default tuning option

%% Data Association Threshold (Mahalanobis Distance)
gating_threshold = 15;  
method = 2; % 1-Euclidean distance   2-Mahalanobis distance

%% Outlier rejection:
% EKF in this application suffers from outliers,
% here we implement the simplest possible solution, if the
% change of the estimation is going to be very large, we reject
% the measurement. In principle, this feature should not be used.
max_update = 1e1; % maximum allowed change in the measurement update

%% Flag indicating whether landmark has been augmented into state
%hasLandmark = false; %% non-augmented initially

%% Main Loop 
for i = 1:nSteps
    % Get current timestamp
    time = time + dt;
    % Get the control input
    u = doControl(time);
    % we assume that the control input is affected by an unknown disturbance
    % simulated as a white noise:
    ureal = u + sqrt(Qsigma)*randn(2,1);
    
    % True (ground) state update (nonlinear motion model)
    xGnd = xGnd + [ dt*cos(xGnd(3))*ureal(1); dt*sin(xGnd(3))*ureal(1); dt*ureal(2)];
                
    % Simulate Observation
    % we will generate a variable z where: 
    % the first column is the measured distance to the landmark
    % the second column is the measured bearing to the landmark
    % the third column is the true x-position of the landmark
    % the fourth column is the true y-position of the landmark
    z = [];
    if No_of_Landmarks > 0
        for iz = 1:size(landMarks,1)
            % Range measurement
            d = norm(xGnd(1:2)' - landMarks(iz,:));
            % Bearing measurement
            alpha = atan2(landMarks(iz,2) - xGnd(2), landMarks(iz,1) - xGnd(1)) - xGnd(3);
            if d < MAX_RANGE % If landmark in the range of measurement
                % Add measurement noise 
                z = [z; [ max(0, d + sqrt(Rsigma(1,1))*randn), alpha + sqrt(Rsigma(2,2))*randn, landMarks(iz,:)]];
            end
        end
    end

    % The odometry position is generated by the deterministic control action u:
    xOdom = xOdom + [ dt*cos(xOdom(3))*u(1); dt*sin(xOdom(3))*u(1); dt*u(2) ]; % Dead_reckoning
                  
    %% Motion (Prediction) Update
    n_states = length(xEst);
    % Linearisition
    A_rb = [ 1, 0, -u(1)*sin(xEst(3));
              0, 1,  u(1)*cos(xEst(3));
              0, 0,  1 ];
    B_rb = [ cos(xEst(3)), 0;
              sin(xEst(3)), 0;
              0,            1 ];
    % Full states A
    A = eye(n_states);
    A(1:3,1:3) = A_rb;

    % Full states B
    B = zeros(n_states,2);
    B(1:3,:) = B_rb;

    P = A * P * A' + B * Qtune * B'; % time evolution of the covariance
    % nonlinear evolution is the same as the odometry evolution:
    xEst(1:3) = xEst(1:3) + [dt*cos(xEst(3))*u(1); dt*sin(xEst(3))*u(1); dt*u(2)];

    %% Measurement Update
    if ~isempty(z)
        for iz = 1:size(z,1)
            d_meas = z(iz,1);
            bearing_meas = z(iz,2);

            % data association
            % if associated, update the state (i_lm>0)
            % if new, augment the state (i_lm=0)
            i_lm =  dataAssociation(xEst, P, d_meas, bearing_meas, Rtune, gating_threshold, method);

            if i_lm == 0 % if new
                 n_states = length(xEst); % current number of states
                 lm_x = xEst(1) + d_meas*cos(xEst(3)+bearing_meas);
                 lm_y = xEst(2) + d_meas*sin(xEst(3)+bearing_meas);
                 xEst = [xEst; lm_x; lm_y];

                % Jacobians for augmentation:
                % G_v: derivative of landmark position w.r.t. robot state (2x3)
                 G_v = [ 1, 0, -d_meas*sin(xEst(3) + bearing_meas);
                         0, 1,  d_meas*cos(xEst(3) + bearing_meas) ];
                % G_z: derivative of landmark position w.r.t. measurement (2x2)
                 G_z = [ cos(xEst(3) + bearing_meas), -d_meas*sin(xEst(3) + bearing_meas);
                         sin(xEst(3) + bearing_meas),  d_meas*cos(xEst(3) + bearing_meas) ];

                 % Augment landmark covariance:
                 P_xx = P(1:3, 1:3); % robot itself
                 P_xm = P(1:3, 4:n_states);

                 P_mm_1 = G_v*P_xx*G_v' + G_z*Rtune*G_z';
                 P_mm_2 = G_v*P_xm;
                 P_mm_3 = P_mm_2';
                 P_xm_1 = P_xx * G_v';  % 3x2
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

            else % if old
                n_states = length(xEst); % current number of states
                % i_lm is the index of landmark in the state
                id_x = 3 + 2*(i_lm-1) + 1; 
                id_y = id_x + 1;

                % innovation
                dx = xEst(id_x) - xEst(1); 
                dy = xEst(id_y) - xEst(2);
                r_pred = sqrt(dx^2 + dy^2);
                bearing_pred = wrapToPi(atan2(dy, dx) - xEst(3));

                y_innov = [d_meas - r_pred;
                           wrapToPi(bearing_meas - bearing_pred)];

                % jacobian
                H_rb = [ -dx/r_pred, -dy/r_pred, 0;
                         dy/(r_pred^2), -dx/(r_pred^2), -1];
                H_l = [ dx/r_pred, dy/r_pred;
                       -dy/(r_pred^2), dx/(r_pred^2)];

                H = zeros(2, n_states);
                H(:,1:3) = H_rb; % robot part
                H(:,id_x:id_y) = H_l;

                % innovation covariance
                S = H * P * H' + Rtune;
                % Kalman gain
                K = P * H' / S;

                % outlier rejection
                if norm(K * y_innov) < max_update
                    xEst = xEst + K * y_innov;
                    P = (eye(n_states) - K * H) * P;
                else
                    disp(['Rejected measurement at time ', num2str(time)]);
                end
                % xEst(3) = wrapToPi(xEst(3));
            end
        end
    end
    
    % Record error (position error between ground truth and estimated robot position)
    errs = [errs, norm(xGnd(1:2)' - xEst(1:2)')];
    
%% Plotting 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%     Save Data and Plot     %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save simulation results
    localizer.time = [localizer.time; time];
    localizer.xGnd = [localizer.xGnd; xGnd'];
    localizer.xOdom = [localizer.xOdom; xOdom'];
    localizer.xEst = [localizer.xEst; xEst(1:3)'];
    
   % Animation (update every 10 steps)
    if rem(i,10)==0 
        hold off;
        plot(localizer.xGnd(:,1), localizer.xGnd(:,2), '.b'); hold on;
        if No_of_Landmarks > 0
            plot(landMarks(:,1), landMarks(:,2), 'pk', 'MarkerSize',10); hold on;
        end
        if ~isempty(z)
            for iz = 1:size(z,1)
                ray = [ xGnd(1:2)'; z(iz,3:4) ];
                plot(ray(:,1), ray(:,2), '-r'); hold on;
            end
        end
        plot(localizer.xOdom(:,1), localizer.xOdom(:,2), '.k'); hold on;
        plot(localizer.xEst(:,1), localizer.xEst(:,2), '.r'); hold on;
        
        % Plot real-time estimated landmarks (if any)
        if length(xEst) > 3
            lm_est = reshape(xEst(4:end), 2, [])';
            plot(lm_est(:,1), lm_est(:,2), 'mo', 'MarkerSize',8, 'LineWidth',2); hold on;
        end
        
        axis equal;
        grid on;
        drawnow;
    end
    
    % Record landmark estimates history (only if exactly 2 landmarks are present)
    n_lm = (length(xEst)-3)/2;
    if n_lm == 2
        lm_est = reshape(xEst(4:end), 2, [])';
    else
        lm_est = [NaN, NaN; NaN, NaN];
    end
    if ~isfield(localizer, 'landmarkEstHist')
        localizer.landmarkEstHist = [];
    end
    % Append as a row: [lm1_x, lm1_y, lm2_x, lm2_y]
    localizer.landmarkEstHist = [localizer.landmarkEstHist; lm_est(:)'];
    
    pause(0.01)
    
end
%% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%    END    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save final estimated landmarks to localizer for final result plotting
n_states = length(xEst);
if n_states > 3
    localizer.landmarkEst = reshape(xEst(4:end), 2, [])';
end
localizer.landmarkTrue = landMarks;

% Draw final results (robot trajectory, true landmarks, final estimated landmarks)
drawResults(localizer);
toc;
disp(num2str(sum(errs.^2)));

figure(2)
plot(0.1:dt:endTime, errs)
xlabel('Time (s)')
ylabel('Robot Position Error (m)')
xlim([0, endTime]) 

%% Landmark Update Magnitude Plot (only if exactly 2 landmarks are estimated)
n_est_landmarks = (length(xEst)-3)/2;
if n_est_landmarks == 2
    times = localizer.time;
    M = length(times);
    update1 = zeros(M-1,1);
    update2 = zeros(M-1,1);
    for j = 2:M
        lm1_prev = localizer.landmarkEstHist(j-1, 1:2);
        lm1_curr = localizer.landmarkEstHist(j, 1:2);
        update1(j-1) = norm(lm1_curr - lm1_prev);
        
        lm2_prev = localizer.landmarkEstHist(j-1, 3:4);
        lm2_curr = localizer.landmarkEstHist(j, 3:4);
        update2(j-1) = norm(lm2_curr - lm2_prev);
    end
    figure(3)
    plot(times(2:end), update1, 'b-', 'LineWidth',2); hold on;
    plot(times(2:end), update2, 'r-', 'LineWidth',2); hold off;
    xlim([0, endTime])  
    xlabel('Time (s)')
    ylabel('Update Magnitude (m)')
    legend('Landmark 1 Update','Landmark 2 Update')
    title('Estimation Update Magnitude for Landmarks')
else
    
    disp('number of stated landmarks bigger than 2 ');
end

end

%% Other functions

% degree to radian
function radian = toRadian(degree)
    radian = degree/180*pi;
end

function [] = drawResults(localizer)
% Plot results

    figure(1);
    hold off;
    x = [ localizer.xGnd(:,1:2) localizer.xEst(:,1:2)];
    set(gca, 'fontsize', 12, 'fontname', 'times');
    plot(x(:,1), x(:,2), '-.b', 'linewidth', 2); hold on;
    plot(x(:,3), x(:,4), '-r', 'linewidth', 1); hold on;
    plot(localizer.xOdom(:,1), localizer.xOdom(:,2), '--k', 'linewidth', 2); hold on;
    % plot true landmark position
    if isfield(localizer, 'landmarkTrue') && ~isempty(localizer.landmarkTrue)
        plot(localizer.landmarkTrue(:,1), localizer.landmarkTrue(:,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','k'); hold on;
    end
    % Plot final estimated landmarks 
    if isfield(localizer, 'landmarkEst') && ~isempty(localizer.landmarkEst)
        plot(localizer.landmarkEst(:,1), localizer.landmarkEst(:,2), 'mo', 'MarkerSize',12, 'LineWidth',2); hold on;
    end
    title('Localization Result', 'fontsize', 12, 'fontname', 'times');
    xlabel('X (m)', 'fontsize', 12, 'fontname', 'times');
    ylabel('Y (m)', 'fontsize', 12, 'fontname', 'times');
    legend('Ground Truth', 'Extended Kalman Filter', 'Odometry Only');
    grid on;
    axis equal;
end

function [ u ] = doControl( time )
    % The input has been designed to increase smoothly until it achieves the
    % steady state values below.
    % Calc Input Parameter
    T = 10; % [sec]
    % [V yawrate]
    V = 0.1; % [m/s]
    yawrate = 5; % [deg/s]
    u = [ V*(1-exp(-time/T)), toRadian(yawrate)*(1-exp(-time/T)) ]';
end

function idx_landmark = dataAssociation(xEst, P, r_meas, bearing_meas, R, gating_threshold, method)
% Default method is Mahalanobis distance
if nargin < 7
    method = 2;
end

% number of states
n_state = length(xEst);
n_landmarks = (n_state - 3) / 2; % number of landmarks in the state

% If no landmarks exist, return new
if n_landmarks == 0
    idx_landmark = 0;
    return;
end

% Extract the robot state 
rx = xEst(1);
ry = xEst(2);
rtheta = xEst(3);

% actual measurement 
z_actual = [r_meas; bearing_meas];

% store distances for each landmark
dists = zeros(n_landmarks, 1);

for i = 1:n_landmarks
    % index of the i-th landmark in the state vector
    idxX = 3 + 2*(i-1) + 1;
    idxY = idxX + 1;
    lm_x = xEst(idxX);
    lm_y = xEst(idxY);
    
    % geometric relation between robot and landmark
    dx = lm_x - rx;
    dy = lm_y - ry;
    pred_range = sqrt(dx^2 + dy^2);
    pred_bearing = wrapToPi(atan2(dy, dx) - rtheta);
    
    % innovation
    z_pred = [pred_range; pred_bearing];
    innov = z_actual - z_pred;
    innov(2) = wrapToPi(innov(2));  % Normalize the bearing difference
    
    % Compute the distance
    switch method
        case 1  % Euclidean distance
            dists(i) = norm(innov);
        case 2  % Mahalanobis distance
            % Jacobian 
            H_r = [ -dx/pred_range, -dy/pred_range, 0;
                     dy/(pred_range^2), -dx/(pred_range^2), -1];
            H_l = [ dx/pred_range, dy/pred_range;
                   -dy/(pred_range^2), dx/(pred_range^2)];
            % full Jacobian matrix
            H = zeros(2, n_state);
            H(:,1:3) = H_r;
            H(:, idxX:idxY) = H_l;
            
            % innovation covariance matrix S
            S = H * P * H' + R;
            % Calculate the Mahalanobis distance (square)
            dists(i) = innov' / S * innov;
        otherwise
            error('Unknown method parameter. Please choose 1 (Euclidean) or 2 (Mahalanobis).');
    end
end

% landmark with the minimum distance
[dist_min, i_min] = min(dists);

% If the minimum distance exceeds the threshold, consider it as a new landmark.
if dist_min > gating_threshold
    idx_landmark = 0;
else
    idx_landmark = i_min;
end

end

function setup_localizer()
    global localizer;
    localizer.time = [];                                                       % all historical timestamps
    localizer.xEst = [];                                                       % all estimate results
    localizer.xGnd = [];                                                       % all ground true
    localizer.xOdom = [];                                                      % all odometry only results
    localizer.z = [];                                                          % 
    localizer.PEst = [];
    localizer.u = [];
    localizer.landmarkEst = [];  % estimated landmark position
    localizer.landmarkTrue = []; % true landmark position
end
