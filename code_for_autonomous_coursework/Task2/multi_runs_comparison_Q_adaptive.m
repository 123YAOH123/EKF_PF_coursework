clear; clc;

%% Parameters
numRuns   = 1000;    % How many times to repeat the experiment
endTime   = 50;      % Simulation duration (seconds)
dt        = 0.1;     % Time step
No_of_LMs = 20;       % Number of landmarks
worldsize = 1.5;     % For generating random landmark positions
Qsigma = diag([0.1, 0.01]).^2;  % "true" process noise 
Rsigma = diag([0.01, 0.01]).^2; % measurement noise
MAX_RANGE = 1;                  % Max range for observation
alpha = 0.95;                   % For Q-adaptive smoothing
initialQturn = Qsigma*0.01^2;  % Qturn

%% Storage of results
allRMSE_adaptive    = zeros(numRuns,1);
allRMSE_nonAdaptive = zeros(numRuns,1);

%% Main loop over multiple runs
for iRun = 1:numRuns

    % ====== 1) Generate random landmarks ======
    rng('shuffle');  % random seed
    landMarks = zeros(No_of_LMs,2);
    for iL = 1:No_of_LMs
        landMarks(iL,:) = [ -worldsize + 2*worldsize*rand, ...
                            -0.5       + 2*worldsize*rand ];
    end

    % ====== 2) Run EKF with Q-adaptive ======
    doAdaptive = true;
    rmseAdp = runOneExperiment(endTime, dt, landMarks, doAdaptive, ...
                               initialQturn, alpha, Qsigma, Rsigma, MAX_RANGE);

    % ====== 3) Run EKF with fixed Q ======
    doAdaptive = false;
    rmseNoAdp = runOneExperiment(endTime, dt, landMarks, doAdaptive, ...
                                 initialQturn, alpha, Qsigma, Rsigma, MAX_RANGE);

    % Store results
    allRMSE_adaptive(iRun)    = rmseAdp;
    allRMSE_nonAdaptive(iRun) = rmseNoAdp;

    fprintf('Run %02d =>  adaptive=%.4f   fixedQ=%.4f\n', iRun, rmseAdp, rmseNoAdp);
end

%% Final average results
avgAdp   = mean(allRMSE_adaptive);
avgNoAdp = mean(allRMSE_nonAdaptive);

fprintf('\n==================================================\n');
fprintf('After %d runs:\n', numRuns);
fprintf('Average RMSE (Adaptive Q) = %.4f\n',   avgAdp);
fprintf('Average RMSE (Fixed   Q) = %.4f\n',   avgNoAdp);


function rmseVal = runOneExperiment(endTime, dt, landMarks, doAdaptive, ...
                                    Qturn, alpha, Qsigma, Rsigma, MAX_RANGE)
    %% Initialize
    time    = 0;
    xEst    = [0; 0; 0];
    xGnd    = [0; 0; 0];
    xOdom   = [0; 0; 0];
    P       = zeros(3,3);
    nSteps  = ceil(endTime/dt);

    errs = [];  % store error at each step

    for step = 1:nSteps

        time = time + dt;

        % 1) Control input (no noise)
        u = doControl(time);

        % 2) Add "true" process noise for ground truth evolution
        ureal = u + sqrt(Qsigma)*randn(2,1);

        %   Evolve ground truth
        xGnd = xGnd + [ dt*cos(xGnd(3))*ureal(1);
                        dt*sin(xGnd(3))*ureal(1);
                        dt*ureal(2)];

        % 3) Generate observations (landmarks)
        z = [];
        for iLm = 1:size(landMarks,1)
            d = norm(xGnd(1:2)' - landMarks(iLm,:));
            if d < MAX_RANGE
                realAngle = atan2(landMarks(iLm,2)-xGnd(2), ...
                                  landMarks(iLm,1)-xGnd(1)) - xGnd(3);
                measDist = d + sqrt(Rsigma(1,1))*randn;
                measAng  = realAngle + sqrt(Rsigma(2,2))*randn;
                z = [z; [measDist, measAng, landMarks(iLm,1), landMarks(iLm,2)]];
            end
        end

        % 4) Odometry (ideal)
        xOdom = xOdom + [ dt*cos(xOdom(3))*u(1);
                          dt*sin(xOdom(3))*u(1);
                          dt*u(2) ];

        % 5) EKF predict
        A = [1, 0, -u(1)*sin(xEst(3));
             0, 1,  u(1)*cos(xEst(3));
             0, 0,  1];
        B = [cos(xEst(3)), 0;
             sin(xEst(3)), 0;
             0,            1];

        %   Covariance prediction
        P = A*P*A' + B*Qturn*B';

        %   State prediction
        xPred = xEst + [ dt*cos(xEst(3))*u(1);
                         dt*sin(xEst(3))*u(1);
                         dt*u(2) ];
        xEst  = xPred;

        % 6) Measurement update
        if ~isempty(z)
            max_update = 1e1;
            for iz = 1:size(z,1)
                % predicted distance to this landmark
                r = norm(xEst(1:2)' - z(iz,3:4));
                if r>0.1
                    alpha_pred = atan2(z(iz,4)-xEst(2), z(iz,3)-xEst(1)) - xEst(3);
                    y = [ z(iz,1) - r;
                          wrapToPi( z(iz,2) - alpha_pred ) ];
                    H = [ (xEst(1)-z(iz,3))/r,   (xEst(2)-z(iz,4))/r,   0;
                          (z(iz,4)-xEst(2))/(r^2), -(z(iz,3)-xEst(1))/(r^2), -1 ];

                    S = H*P*H' + Rsigma;
                    K = P*H'/S;

                    if norm(K*y) < max_update
                        xEst = xEst + K*y;
                        P = (eye(3)-K*H)*P;
                    end
                end
            end
        end

        % 7) Adaptive Q
        if doAdaptive
            dx = xEst - xPred;
            Qturn = alpha*Qturn + (1-alpha)*((pinv(B)*dx)*(pinv(B)*dx)');
        end

        % 8) Log position error
        errNow = norm( xGnd(1:2)' - xEst(1:2)' );
        errs   = [errs, errNow];
    end

    % 9) Final RMSE
    rmseVal = sqrt(mean(errs.^2));
end

function out = doControl(t)
    T = 10; 
    V = 0.1;
    yawDeg = 5;
    yawRad = yawDeg*pi/180;
    out = [ V*(1-exp(-t/T));
            yawRad*(1-exp(-t/T)) ];
end

function angleOut = wrapToPi(angleIn)
    angleOut = mod(angleIn+pi, 2*pi)-pi;
end
