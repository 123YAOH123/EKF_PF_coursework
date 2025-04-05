%% Parameter Settings
No_of_Landmarks = 20;       % Number of landmarks (can be adjusted as needed)
worldsize = 1.5;            % Defines x-axis range as [-worldsize, worldsize]
m = 1000;                   % Number of Monte Carlo trials
n = 10000;                    % Number of particles per trial

% Array to store the suitable radar detection range for each trial
range_values = zeros(m,1);

%% Begin Monte Carlo Simulation
for j = 1:m
    % Randomly generate landmark positions
    % x in [-worldsize, worldsize], y in [-0.5, 2*worldsize - 0.5]
    landMarks = zeros(No_of_Landmarks, 2);
    for i = 1:No_of_Landmarks
        landMarks(i,:) = [-worldsize + 2*worldsize*rand, -0.5 + 2*worldsize*rand];
    end

    % Randomly generate particles within the same area as landmarks
    particles = [ -worldsize + 2*worldsize*rand(n,1), ...
                  -0.5 + 2*worldsize*rand(n,1) ];

    % For each particle, compute its distance to all landmarks
    % and find the distance to the nearest landmark
    min_distances = zeros(n,1);
    for k = 1:n
        distances = sqrt(sum((landMarks - particles(k,:)).^2, 2));
        min_distances(k) = min(distances);
    end

    % For this trial, the largest of the minimum distances is
    % considered the suitable radar detection range
    range_values(j) = max(min_distances);
end

%% Compute the average suitable detection range across all trials
suitable_range = mean(range_values);
disp(['Suitable radar detection range = ', num2str(suitable_range)]);
