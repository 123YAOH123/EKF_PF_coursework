function [] = particle_filter_localization()
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
xEst=[0 0 0]';                                                             % Estimated State [x y yaw]'
xGnd = xEst;                                                               % GroundTruth State 
xOdom = xGnd;                                                              % Odometry-only = Dead Reckoning 

%% real noises and sensor range 
Qsigma=diag([0.1 0.01]).^2;  % prediction model
Rsigma=diag([0.1 0.01]).^2;     % observation model
MAX_RANGE =1; % longest lidar observation confined

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%    Landmarks and particles      %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

%% landmark positions


No_of_Landmarks =20; % select the number of landmark         
worldsize = 1.5; % select the size of your landmark
landMarks = [-worldsize+2*worldsize*rand(No_of_Landmarks,1), -0.5+2*worldsize*rand(No_of_Landmarks,1)];

% Last marks are saved in case you want to reuse them:
%save('lastlandmarks.mat', 'landMarks'); 

% load('lastlandmarks')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%        Filter setup             %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Num of particles, initialized
NP = 50;
Qtune=Qsigma;    % Covariance Matrix for predict model
Rtune=Rsigma;    % Covariance Matrix for measurements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  


%% Initialisation
% particles produced
px=repmat(xEst, 1, NP);
% weights of particles produced
pw=zeros(1, NP) + 1 / NP;

% sum of error (used for plot)
errs=[];
setup_localizer()

%% Main Loop 
for i=1 : nSteps
    % Get current timestamp
    time = time + dt;
    % Get the control input
    u=doControl(time);
    % Get observation results
    [z,xGnd,xOdom]=doObservation(xGnd, xOdom, u, landMarks, MAX_RANGE,time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%      Propgation and Update      %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % Process every particle
    for ip=1:NP
        x=px(:,ip);                                                        % Pose of this particle 
        w=pw(ip);                                                          % Weight of this particle

        usim= u + [sqrt(Qtune(1,1))*randn; sqrt(Qtune(2,2))*randn]; 

        J = [ dt*cos(x(3))  0
              dt*sin(x(3))  0
              0             dt];

        x = x + J*usim;



        % z takes the from of distance measurement and the position of the
        % landmark such that 
        % z = [ dist to landMark , angle to landMark, landmark x, landmank y; ... ]

        % Update the weight for each landmark measurement
        if ~isempty(z)
            for iz=1:length(z(:,1))
                % Distanct from robot to landmark from the observation model
                 pz=norm(x(1:2)'-z(iz,3:4));
                 alpha = atan2(-x(2) +z(iz,4), -x(1) +z(iz,3)) - x(3);
    
                % Difference between  the distance generated from the
                % observation model and the measured distance z(i,1)
                 dz=pz-z(iz,1);
                 dalpha=wrapToPi(alpha-z(iz,2));
                
                % Update the weighting using a Gaussian function  
                w = w * 1/sqrt(2*pi*Rtune(1,1))*exp(-1/2*dz^2/Rtune(1,1));
                w = w * 1/sqrt(2*pi*Rtune(2,2))*exp(-1/2*dalpha^2/Rtune(2,2));
            end
        end
        px(:,ip)=x;
        pw(ip)=w;       
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%           Resampling            %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    pw = pw./sum(pw); %Normalisation
    % Given px being an array of robot poses [x, y, theta]'
    % and the weighting of each particle

    % implement a resampling algorithm
    [px, xEst] = systematic_resampling(px, pw);
   
    
    % Reset weights to uniform:
    pw = ones(1, NP) / NP;
    % Error computation:
    errs=[errs, norm(xGnd(1:2)'-xEst(1:2)')];

%% Plotting 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%     Save Data and Plot     %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Simulation Result
    localizer.time=[localizer.time; time];
    localizer.xGnd=[localizer.xGnd; xGnd'];
    localizer.xOdom=[localizer.xOdom; xOdom'];
    localizer.xEst=[localizer.xEst;xEst'];
    localizer.u=[localizer.u; u'];
    
    % Animation (remove some flames)
    if rem(i,10)==0 
        hold off;
        arrow=0.5;
        for ip=1:NP
            quiver(px(1,ip),px(2,ip),arrow*cos(px(3,ip)),arrow*sin(px(3,ip)),'ok');hold on;
        end
        plot(localizer.xGnd(:,1),localizer.xGnd(:,2),'.b');hold on;
        plot(landMarks(:,1),landMarks(:,2),'pk','MarkerSize',10);hold on;
        if~isempty(z)
            for iz=1:length(z(:,1))
                ray=[xGnd(1:2)';z(iz,3:4)];
                plot(ray(:,1),ray(:,2),'-r');hold on;
            end
        end
        plot(localizer.xOdom(:,1),localizer.xOdom(:,2),'.k');hold on;
        plot(localizer.xEst(:,1),localizer.xEst(:,2),'.r');hold on;
        axis equal;
        grid on;
        drawnow;
    end
    
end
%% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%    END    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% draw the final results of localizer, compared to odometry & ground truth
drawResults(localizer);
toc
disp(['The sum of the squared error is ',num2str(sum(errs.^2))]);
figure(2)
plot(0.1:0.1:endTime,errs)
xlabel("time")
ylabel("error square")
end

%% Other functions
% degree to radian
function radian = toRadian(degree)
    radian = degree/180*pi;
end

function []=drawResults(localizer)
%Plot Result
 
    figure(1);
    hold off;
    x=[ localizer.xGnd(:,1:2) localizer.xEst(:,1:2)];
    set(gca, 'fontsize', 12, 'fontname', 'times');
    plot(x(:,1), x(:,2),'-.b','linewidth', 4); hold on;
    plot(x(:,3), x(:,4),'r','linewidth', 4); hold on;
    plot(localizer.xOdom(:,1), localizer.xOdom(:,2),'--k','linewidth', 4); hold on;
    title('Localization Result', 'fontsize', 12, 'fontname', 'times');
    xlabel('X (m)', 'fontsize', 12, 'fontname', 'times');
    ylabel('Y (m)', 'fontsize', 12, 'fontname', 'times');
    legend('Ground Truth','Particle Filter','Odometry Only');
    grid on;
    axis equal;
end

function [ u ] = doControl( time )
    % The input has been designed increase smoothly until it achieves the
    % steady state values below.
    %Calc Input Parameter
    T=10; % [sec]
    % [V yawrate]
    V=0.1; % [m/s]
    yawrate = 5; % [deg/s]
    u =[ V*(1-exp(-time/T)) toRadian(yawrate)*(1-exp(-time/T))]';
end
 
% do Observation model 
function [z, xGnd, xOdom] = doObservation(xGnd, xOdom, u, landMarks, MAX_RANGE, time)
    global Qsigma;
    global Rsigma;
    global dt
    xOdom=xOdom+[dt*cos(xOdom(3))*u(1);dt*sin(xOdom(3))*u(1) ; dt*u(2)];
    u=u+sqrt(Qsigma)*randn(2,1); % add noise randomly
    xGnd = xGnd+[dt*cos(xGnd(3))*u(1);dt*sin(xGnd(3))*u(1) ; dt*u(2)];
    %Simulate Observation
    z=[];
        for iz = 1:size(landMarks, 1)
        d = norm(xGnd(1:2)' - landMarks(iz, :));
        alpha = atan2(-xGnd(2)' + landMarks(iz,2), -xGnd(1)' + landMarks(iz,1)) - xGnd(3);
        
        if d < MAX_RANGE
            % 
            noisy_d = max(0, d + sqrt(Rsigma(1,1)) * randn);
            noisy_alpha = alpha + sqrt(Rsigma(2,2)) * randn;

            % -------------------
            % Jerky error 
            % -------------------
            if abs(time - 10) < 1e-6 || abs(time - 40) < 1e-6
                noisy_d = noisy_d + 0.5;                        % 
                noisy_alpha = noisy_alpha + deg2rad(20);        % 
            end

            z = [z; [noisy_d, noisy_alpha, landMarks(iz,:)]];  % 
        end
    end
end

function setup_localizer()
    global localizer;
    localizer.time = [];                                                       % all historical timestamps
    localizer.xEst = [];                                                       % all estimate results
    localizer.xGnd = [];                                                       % all ground true
    localizer.xOdom = [];                                                      % all odometry only results
    localizer.z = [];                                                          % 
    localizer.PEst=[];
    localizer.u=[];
end


function [px, xEst] = systematic_resampling(px, pw)
%% Implementation of systematic resampling 
    NP = length(pw);  % Number of particles
    
    % Compute cumulative sum of weights
    cdf = cumsum(pw);
    
    % Generate a single random starting point
    r = rand(1) / NP;
    U = r + (0:NP-1) / NP; % Equally spaced points
    %xEst = px * pw'; 
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
    %pw = pw(indices);
    % Compute estimated state
    xEst = px * pw';   
end


