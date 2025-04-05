function avg_rmse = particle_filter_localization_1000_modified()
%PARTICLE_FILTER_LOCALIZATION_1000_MODIFIED
% This function runs 1000 experiments of PF localization and computes the average RMSE.
% If one experiment yields NaN for RMSE, that result is discarded.
% The original PF localization code by Zhixin Zhang, Josh Bettles, and JC is used as the core.
% Version 1: 13 Mar 2025

num_experiments = 100;
valid_rmse = [];
nan_count = 0;

for exp = 1:num_experiments
    rmse = run_pf_localization();
    if isnan(rmse)
        nan_count = nan_count + 1;
        disp(['Experiment ', num2str(exp), ': rmse is NaN, skipping...'])
    else
        valid_rmse = [valid_rmse; rmse];
    end
end

if isempty(valid_rmse)
    avg_rmse = NaN;
    disp('All experiments produced NaN for RMSE.');
else
    avg_rmse = mean(valid_rmse);
    disp(['Average RMSE over ', num2str(num_experiments - nan_count), ' valid experiments: ', num2str(avg_rmse)]);
end
end

function rmse = run_pf_localization()
%PARTICLE_FILTER_LOCALIZATION
% TASK for Particle Filter localization
% This code has been produced by Zhixin Zhang, Josh Bettles, and JC.
% Version 1: 13 Mar 2025

%% initialization
close all;
clear all;
 
disp('Particle Filter program start!!')
 
tic;
time = 0;
endTime = 50; % second

global dt;
global localizer;
global Qsigma
global Rsigma
dt = 0.1; % second

nSteps = ceil((endTime - time)/dt);
xEst=[0 0 0]';          % Estimated State [x y yaw]'
xGnd = xEst;            % GroundTruth State 
xOdom = xGnd;           % Odometry-only = Dead Reckoning 

%% real noises and sensor range 
Qsigma=diag([0.1 0.01]).^2;  % prediction model
Rsigma=diag([0.1 0.01]).^2;  % observation model
MAX_RANGE = 1; % longest lidar observation confined

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%    Landmarks and particles      %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

%% landmark positions
No_of_Landmarks = 20;       % select the number of landmarks         
worldsize = 1.5;            % select the size of your landmark
landMarks = [-worldsize+2*worldsize*rand(No_of_Landmarks,1), -0.5+2*worldsize*rand(No_of_Landmarks,1)];

% Last marks are saved in case you want to reuse them:
save('lastlandmarks.mat', 'landMarks'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%        Filter setup             %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of particles, initialized
NP = 100;
Qtune = Qsigma;    % Covariance Matrix for predict model
Rtune = Rsigma;    % Covariance Matrix for measurements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

%% Initialisation
% particles produced
px = repmat(xEst, 1, NP);
% weights of particles produced
pw = zeros(1, NP) + 1 / NP;

% sum of error (used for RMSE computation)
errs = [];
setup_localizer();

%% Main Loop 
for i = 1 : nSteps
    % Get current timestamp
    time = time + dt;
    % Get the control input
    u = doControl(time);
    % Get observation results
    [z, xGnd, xOdom] = doObservation(xGnd, xOdom, u, landMarks, MAX_RANGE);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%      Propagation and Update      %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % Process every particle
    for ip = 1:NP
        x = px(:,ip);        % Pose of this particle 
        w = pw(ip);          % Weight of this particle

        usim = u + [sqrt(Qtune(1,1))*randn; sqrt(Qtune(2,2))*randn]; 

        J = [ dt*cos(x(3))  0
              dt*sin(x(3))  0
              0             dt];

        x = x + J*usim;

        % z takes the form of distance measurement and the position of the
        % landmark such that 
        % z = [ distance to landmark, angle to landmark, landmark x, landmark y; ... ]

        % Update the weight for each landmark measurement
        if ~isempty(z)
            for iz = 1:size(z,1)
                % Distance from robot to landmark from the observation model
                pz = norm(x(1:2)' - z(iz,3:4));
                alpha = atan2(-x(2) + z(iz,4), -x(1) + z(iz,3)) - x(3);
    
                % Difference between the predicted distance and the measured distance z(iz,1)
                dz = pz - z(iz,1);
                dalpha = wrapToPi(alpha - z(iz,2));
                
                % Update the weighting using a Gaussian function  
                w = w * 1/sqrt(2*pi*Rtune(1,1)) * exp(-0.5*dz^2/Rtune(1,1));
                w = w * 1/sqrt(2*pi*Rtune(2,2)) * exp(-0.5*dalpha^2/Rtune(2,2));
            end
        end
        px(:,ip) = x;
        pw(ip) = w;       
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%           Resampling            %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    pw = pw ./ sum(pw); % Normalisation

    % implement a resampling algorithm (systematic resampling)
    %[px, xEst] = systematic_resampling(px, pw);
     
     [px, xEst] = improved_systematic_resampling_notune(px, pw);
    % Reset weights to uniform:
    pw = ones(1, NP) / NP;
    % Error computation:
    errs = [errs, norm(xGnd(1:2)' - xEst(1:2)')];

    %%% Plotting code removed for faster running %%%
end
%% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (Plotting code removed for faster running)
toc
% Compute RMSE for this experiment
rmse = sqrt(mean(errs.^2));
end

%% Other functions
% degree to radian
function radian = toRadian(degree)
    radian = degree/180*pi;
end

function [] = drawResults(localizer)
%Plot Result
% This function has been disabled in the fast simulation version.
end

function [ u ] = doControl(time)
    % The input has been designed to increase smoothly until it achieves the
    % steady state values below.
    % Calculate Input Parameter
    T = 10; % [sec]
    % [V, yawrate]
    V = 0.1; % [m/s]
    yawrate = 5; % [deg/s]
    u = [ V*(1-exp(-time/T)), toRadian(yawrate)*(1-exp(-time/T)) ]';
end
 
% do Observation model 
function [z, xGnd, xOdom] = doObservation(xGnd, xOdom, u, landMarks, MAX_RANGE)
    global Qsigma;
    global Rsigma;
    global dt
    xOdom = xOdom + [dt*cos(xOdom(3))*u(1); dt*sin(xOdom(3))*u(1); dt*u(2)];
    u = u + sqrt(Qsigma)*randn(2,1); % add noise randomly
    xGnd = xGnd + [dt*cos(xGnd(3))*u(1); dt*sin(xGnd(3))*u(1); dt*u(2)];
    % Simulate Observation
    z = [];
    for iz = 1:size(landMarks,1)
        d = norm(xGnd(1:2)' - landMarks(iz,:));
        alpha = atan2(-xGnd(2)' + landMarks(iz,2), -xGnd(1)' + landMarks(iz,1)) - xGnd(3);
        if d < MAX_RANGE 
            z = [z; [max(0, d + sqrt(Rsigma(1,1))*randn), alpha + sqrt(Rsigma(2,2))*randn, landMarks(iz,:)]];
        end
    end
end

function setup_localizer()
    global localizer;
    localizer.time = [];      % all historical timestamps
    localizer.xEst = [];      % all estimated results
    localizer.xGnd = [];      % all ground truth states
    localizer.xOdom = [];     % all odometry-only results
    localizer.z = [];         
    localizer.PEst = [];
    localizer.u = [];
end

%%%%%%%%%% 1
function [px, xEst] = systematic_resampling(px, pw)
%% Implementation of systematic resampling 
    NP = length(pw);  % Number of particles
    
    % Compute cumulative sum of weights
    cdf = cumsum(pw);
    
    % Generate a single random starting point
    r = rand(1) / NP;
    U = r + (0:NP-1) / NP; % Equally spaced points
    % Resampling step
    indices = zeros(1, NP);
    i = 1;
    for j = 1:NP
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    % Select new particles based on indices
    px = px(:, indices);
    % Compute estimated state
    xEst = px * pw';
end

function [px, xEst] = multinomial_resampling(px, pw)
%% Implementation of multinomial resampling
    NP = length(pw);                 % Number of particles
    cdf = cumsum(pw);                % Cumulative distribution function
    U = rand(1, NP);                 % Generate NP uniform random numbers
    indices = zeros(1, NP);
    i = 1;
    for j = 1:NP
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    px = px(:, indices);             % Resample particles using selected indices
    xEst = px * pw';                 % Compute state estimate using original weights
end

function [px, xEst] = stratified_resampling(px, pw)
%% Implementation of stratified resampling
    NP = length(pw);                % Number of particles
    cdf = cumsum(pw);               % Cumulative distribution function
    U = ((0:NP-1) + rand(1, NP)) / NP; % Generate stratified random numbers
    indices = zeros(1, NP);
    i = 1;
    for j = 1:NP
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    px = px(:, indices);             % Resample particles using selected indices
    xEst = px * pw';                 % Compute state estimate using original weights
end

%%%%%%%%%% ISR
function [px, xEst] = improved_systematic_resampling_notune(px, pw)
%% Implementation of Improved Systematic Resampling (ISR)
% px: Particles matrix [state_dim x num_particles]
% pw: Weights of the particles (1 x num_particles)

    NP = length(pw);  % Number of particles
    
    % Define tau as the 1st percentile of weights
    tau = prctile(pw, 1); % 1% percentile
    rho = 1e-4; % Small reassigned value for very low weights
    
    % Identify very low weights and assign them a smaller value
    very_low_weights = pw < tau;
    pw(very_low_weights) = rho;
    
    % Normalize the updated weights
    pw = pw / sum(pw);
    
    % Compute cumulative sum of weights
    cdf = cumsum(pw);
    
    % Generate a single random starting point
    r = rand(1) / NP;
    U = r + (0:NP-1) / NP; % Equally spaced points
    
    % Resampling step
    indices = zeros(1, NP);
    i = 1;
    for j = 1:NP
        while U(j) > cdf(i)
            i = i + 1;
        end
        indices(j) = i;
    end
    
    % Select new particles based on indices
    px = px(:, indices);
    
    % Compute estimated state
    xEst = px * pw';   
end



%%%%%%%%%% MSV Resampling
function [pxNew, xEst] = msv_resampling(px, pw, N)
%% MSV (Minimum Sampling Variance) Resampling
% Inputs:
%   px  - [D x M] particle states
%   pw  - [1 x M] normalized weights
%   N   - number of resampled particles
% Outputs:
%   pxNew - [D x N] resampled particles
%   xEst  - estimated state (mean of pxNew)

    if nargin < 3
        N = length(pw);  % default: keep same particle count
    end

    pw = pw(:)' / sum(pw);     % ensure normalized row vector
    M = length(pw);
    pxNew = zeros(size(px,1), N);  % output matrix

    % Step 1: floor(N * w) → integer part
    N_floor = floor(N * pw);       % integer copy count
    v = N * pw - N_floor;          % residual (fractional part)
    L = sum(N_floor);              % number of allocated particles
    R = N - L;                     % remaining particles to assign

    % Step 2: assign R residual particles to Top-R residuals
    if R > 0
        [~, idx] = sort(v, 'descend'); % sort by fractional weight
        N_floor(idx(1:R)) = N_floor(idx(1:R)) + 1;  % add 1 to top R
    end

    % Step 3: generate new particles
    n = 0;
    for m = 1:M
        for h = 1:N_floor(m)
            n = n + 1;
            pxNew(:, n) = px(:, m);
        end
    end

    % Step 4: estimate state using mean (uniform weights)
    xEst = mean(pxNew, 2);
end

