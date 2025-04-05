%% EKF Localization Multi-Experiment Statistics
clear; clc;
%% Parameter Settings
numRuns   = 1000;      % Number of experiments 
endTime   = 50;       % Simulation time [sec]
dt        = 0.1;      % Time step [sec]
nSteps    = ceil(endTime/dt);

% Noise Parameters
Qsigma = diag([0.1, 0.01]).^2;   % Process noise (prediction model)
Rsigma = diag([0.01, 0.01]).^2;   % Measurement noise (observation model)
MAX_RANGE = 1;                  % Maximum range of radar observations

% Landmark Parameters 
No_of_Landmarks = 5;
worldsize = 1.5;

% Preallocate arrays to store metrics for each experiment
allRMSE      = zeros(numRuns,1);
allMaxNoDet  = zeros(numRuns,1);
allFracNoDet = zeros(numRuns,1);

% Inline function: normalize angle to [-pi, pi]
wrapToPi = @(angle) mod(angle+pi,2*pi)-pi;

%% Main Loop for Multiple Experiments
for run = 1:numRuns
    % Generate a new set of landmarks for this experiment
    landMarks = zeros(No_of_Landmarks,2);
    for i = 1:No_of_Landmarks
        landMarks(i,:) = [-worldsize + 2*worldsize*rand, -0.5 + 2*worldsize*rand];
    end
    
    % Initialize state variables
    time = 0;
    xEst = [0;0;0];    % EKF estimated state [x; y; yaw]
    xGnd = [0;0;0];    % Ground truth state
    xOdom = [0;0;0];   % dead reckoning
    P = zeros(3,3);    % Covariance matrix initialization

    errs = [];  % Array to record the position error at each time step

    % Initialize no-detection statistics variables
    consecutiveNoDetection = 0;   % Current consecutive steps with no landmark detection
    maxConsecutiveNoDetection = 0; % Maximum consecutive no-detection steps in this experiment
    totalNoDetectionSteps = 0;     % Total number of steps with no landmark detection

    %% Simulation Loop
    for step = 1:nSteps
        time = time + dt;
        
        %% Calculate Control Input (Smoothly increasing to steady state)
        T = 10;         % Time constant [sec]
        V = 0.1;        % Linear velocity [m/s]
        yawrate_deg = 5;  % Angular velocity [deg/s]
        yawrate = yawrate_deg*pi/180; % Convert to radians
        
        % Control input vector
        u = [ V*(1-exp(-time/T));
              yawrate*(1-exp(-time/T)) ];
          
        % Control input affected by white noise disturbance
        ureal = u + sqrt(Qsigma)*randn(2,1);
        
        %% Ground Truth Update (Nonlinear Motion Model)
        xGnd = xGnd + [dt*cos(xGnd(3))*ureal(1); dt*sin(xGnd(3))*ureal(1); dt*ureal(2)];
                    
        %% Simulate Radar Observations (Generate Measurement z)
        % z: [measured distance, measured angle, landmark true x, landmark true y]
        z = [];
        for j = 1:No_of_Landmarks
            d = norm(xGnd(1:2)' - landMarks(j,:));
            if d < MAX_RANGE
                d_meas = max(0, d + sqrt(Rsigma(1,1))*randn);
                alpha = atan2(landMarks(j,2)-xGnd(2), landMarks(j,1)-xGnd(1)) - xGnd(3);
                alpha_meas = alpha + sqrt(Rsigma(2,2))*randn;
                z = [z; d_meas, alpha_meas, landMarks(j,1), landMarks(j,2)];
            end
        end
        
        %% Odometry Update (Based Solely on Control Input Integration)
        xOdom = xOdom + [dt*cos(xOdom(3))*u(1); dt*sin(xOdom(3))*u(1); dt*u(2)];
                      
        %% EKF Prediction Update
        A = [ 1, 0, -u(1)*sin(xEst(3));
              0, 1,  u(1)*cos(xEst(3));
              0, 0,  1 ];
        B = [ cos(xEst(3)), 0;
              sin(xEst(3)), 0;
              0,            1 ];
        P = A*P*A' + B*Qsigma*B';
        
        % Nonlinear state update (same as odometry update)
        xEst = xEst + [dt*cos(xEst(3))*u(1); dt*sin(xEst(3))*u(1); dt*u(2)];
                    
        %% EKF Measurement Update (Update for Each Landmark)
        max_update = 1e1;  % Outlier rejection threshold

        if ~isempty(z)
            % If landmarks are detected at this step, reset consecutive no-detection counter
            consecutiveNoDetection = 0;
            for j = 1:size(z,1)
                r = norm(xEst(1:2)' - z(j,3:4));
                if r > 0.1  % Avoid numerical issues when too close to a landmark
                    alpha_est = atan2(z(j,4)-xEst(2), z(j,3)-xEst(1)) - xEst(3);
                    y = [ z(j,1) - r;
                          wrapToPi(z(j,2) - alpha_est) ];
                    H = [ (xEst(1)-z(j,3))/r, (xEst(2)-z(j,4))/r, 0;
                          (z(j,4)-xEst(2))/(r^2), -(z(j,3)-xEst(1))/(r^2), -1 ];
                    S = H*P*H' + Rsigma;
                    K = P*H'/S;
                    if norm(K*y) < max_update
                        xEst = xEst + K*y;
                        P = (eye(3)-K*H)*P;
                    else
                        disp(['Rejected measurement at time ', num2str(time)]);
                    end
                end
            end
        else
            % If no landmarks are detected at this step, update no-detection statistics
            consecutiveNoDetection = consecutiveNoDetection + 1;
            totalNoDetectionSteps = totalNoDetectionSteps + 1;
            if consecutiveNoDetection > maxConsecutiveNoDetection
                maxConsecutiveNoDetection = consecutiveNoDetection;
            end
        end
        
        %% Record the Position Error for This Time Step
        errs = [errs, norm(xGnd(1:2)' - xEst(1:2)')];
    end  % End of Simulation Loop
    
    %% Compute Metrics for This Experiment
    rmseFinal = sqrt(mean(errs.^2));
    maxNoDetectionTime = maxConsecutiveNoDetection * dt;
    fractionNoDetection = (totalNoDetectionSteps * dt) / (nSteps * dt);
    
    % Store the results for this experiment
    allRMSE(run)      = rmseFinal;
    allMaxNoDet(run)  = maxNoDetectionTime;
    allFracNoDet(run) = fractionNoDetection;
    
    fprintf('Run %d: RMSE = %.4f, MaxNoDetTime = %.4f s, FracNoDet = %.4f\n',...
            run, rmseFinal, maxNoDetectionTime, fractionNoDetection);
end  % End of Multiple Experiments Loop

%% Compute and Display Average Metrics Over All Experiments
avgRMSE     = mean(allRMSE);
avgMaxNoDet = mean(allMaxNoDet);
avgFracNoDet = mean(allFracNoDet);

disp('----------------------------------------');
disp(['After ', num2str(numRuns), ' runs:']);
disp(['Average RMSE = ', num2str(avgRMSE)]);
disp(['Average Maximum No-Detection Time = ', num2str(avgMaxNoDet), ' seconds']);
disp(['Average Fraction of No-Detection Time = ', num2str(avgFracNoDet)]);
varRMSE = var(allRMSE);
disp(['RMSE Variance = ', num2str(varRMSE)]);