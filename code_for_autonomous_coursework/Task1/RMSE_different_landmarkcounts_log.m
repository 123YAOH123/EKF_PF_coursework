%% EKF Localization RMSE vs Landmark Count
clear; clc;

% Landmark counts
landmark_range = 0:5:100;
numRuns = 100; 
rmse_array = zeros(length(landmark_range), 1); 

% 
endTime = 50;
dt = 0.1;
nSteps = ceil(endTime/dt);
Qsigma = diag([0.1, 0.01]).^2;
Rsigma = diag([0.01, 0.01]).^2;
MAX_RANGE = 1;
worldsize = 1.5;
wrapToPi = @(angle) mod(angle+pi,2*pi)-pi;


for idx = 1:length(landmark_range)
    No_of_Landmarks = landmark_range(idx);
    allRMSE = zeros(numRuns,1);

    for run = 1:numRuns
        landMarks = [-worldsize + 2*worldsize*rand(No_of_Landmarks,1), ...
                     -0.5 + 2*worldsize*rand(No_of_Landmarks,1)];
        xEst = [0;0;0]; xGnd = [0;0;0]; xOdom = [0;0;0];
        P = zeros(3,3);
        errs = [];

        for step = 1:nSteps
            time = step * dt;
            T = 10; V = 0.1; yawrate = 5*pi/180;
            u = [ V*(1-exp(-time/T));
                  yawrate*(1-exp(-time/T)) ];
            ureal = u + sqrt(Qsigma)*randn(2,1);
            xGnd = xGnd + [dt*cos(xGnd(3))*ureal(1); dt*sin(xGnd(3))*ureal(1); dt*ureal(2)];

            % Generate measurements
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

            % Prediction
            A = [ 1, 0, -u(1)*sin(xEst(3));
                  0, 1,  u(1)*cos(xEst(3));
                  0, 0,  1 ];
            B = [ cos(xEst(3)), 0;
                  sin(xEst(3)), 0;
                  0,            1 ];
            P = A*P*A' + B*Qsigma*B';
            xEst = xEst + [dt*cos(xEst(3))*u(1); dt*sin(xEst(3))*u(1); dt*u(2)];

            % Update
            if ~isempty(z)
                for j = 1:size(z,1)
                    r = norm(xEst(1:2)' - z(j,3:4));
                    if r > 0.1
                        alpha_est = atan2(z(j,4)-xEst(2), z(j,3)-xEst(1)) - xEst(3);
                        y = [ z(j,1) - r;
                              wrapToPi(z(j,2) - alpha_est) ];
                        H = [ (xEst(1)-z(j,3))/r, (xEst(2)-z(j,4))/r, 0;
                              (z(j,4)-xEst(2))/(r^2), -(z(j,3)-xEst(1))/(r^2), -1 ];
                        S = H*P*H' + Rsigma;
                        K = P*H'/S;
                        if norm(K*y) < 10
                            xEst = xEst + K*y;
                            P = (eye(3)-K*H)*P;
                        end
                    end
                end
            end

            % Error
            errs = [errs, norm(xGnd(1:2)' - xEst(1:2)')];
        end
        allRMSE(run) = sqrt(mean(errs.^2));
    end
    rmse_array(idx) = mean(allRMSE);
    fprintf('Landmarks %d: RMSE = %.4f\n', No_of_Landmarks, rmse_array(idx));
end

% Plot
figure('Position', [100, 100, 800, 400]);  
semilogx(landmark_range, rmse_array, '-o', 'LineWidth', 2);
xlabel('Number of Landmarks (log scale)');
ylabel('Average RMSE');
title('EKF RMSE vs Landmark Count (Log Scale)');
grid on;
