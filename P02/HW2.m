N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]; % Node counts
t_serial = [0.000038, 0.000042, 0.000104, 0.000139, 0.00027, 0.001047, 0.002796, 0.001505, 0.004302, 0.011392, 0.00778, 0.02355, 0.050575, 0.513107, 3.878562, 11.018, 22.05728, 79.138963, 467.492164]; % Serial runtime (seconds)

t_parallel = [ % Parallel runtimes for different thread counts
    0.000161, 0.00039, 0.000325, 0.000603, 0.000507, 0.000808, 0.001026, 0.002729, 0.003647, 0.00757, 0.014713, 0.028001, 0.089096, 1.116179, 1.52396, 4.691029, 17.853263, 96.326148, 646.947696; % n = 2
    0.000445, 0.000895, 0.003867, 0.000739, 0.000862, 0.001391, 0.001874, 0.006123, 0.003257, 0.012219, 0.014551, 0.033892, 0.139956, 0.630356, 2.177106, 6.616843, 20.351811, 80.154131, 636.904578; % n = 4
    0.000815, 0.00056, 0.000647, 0.001178, 0.00292, 0.001393, 0.001801, 0.005489, 0.0042, 0.007649, 0.014784, 0.026488, 0.115834, 0.576149, 3.318689, 9.754526, 23.157662, 83.627556, 704.992624; % n = 8
    0.001169, 0.000879, 0.001826, 0.003696, 0.01606, 0.009878, 0.003877, 0.005378, 0.025118, 0.00745, 0.029557, 0.072855, 0.063143, 0.447809, 3.402987, 13.813126, 32.680899, 100.585141, 661.449287;  % n = 16
    0.005228, 0.003858, 0.013687, 0.013013, 0.015236, 0.029769, 0.013681, 0.018085, 0.033363, 0.04451, 0.03393, 0.038374, 0.111952, 0.578477, 3.052137, 6.48275, 32.226204, 187.262787, 903.02702 % n = 32
];

threads = [2, 4, 8, 16, 32]; % Thread counts
colors = lines(length(threads)); % Generate distinct colors for each thread count

figure;
hold on;

% Plot serial result in black
plot(N, t_serial, 'k', 'LineWidth', 2, 'DisplayName', 'Serial');

% Plot parallel results with different colors
for i = 1:length(threads)
    plot(N, t_parallel(i, :), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', sprintf('n = %d', threads(i)));
end

hold off;
legend('Location', 'NorthWest');
xlabel('Node Count (N)');
ylabel('Runtime (Seconds)');
title('Runtime vs. Node Count for Serial and Parallel Execution');
grid on;

% Customize axis ticks to match power-of-2 spacing
set(gca, 'XTick', 2.^(0:18)); % Set X-ticks to powers of 2 from 2^0 to 2^18
set(gca, 'XTickLabel', {'2^0', '2^1', '2^2', '2^3', '2^4', '2^5', '2^6', '2^7', '2^8', '2^9', '2^10', '2^11', '2^12', '2^13', '2^14', '2^15', '2^16', '2^17', '2^18'}); % Set X-tick labels

% Uncommend the line below for better visualization 
% set(gca, 'XScale', 'log'); % Set X-axis scale to logarithmic if needed for better visualization

%{
N = [2048, 4096, 8192, 16384, 32768, 65536, 131072]; % Node counts
t_serial = [0.02355, 0.050575, 0.513107, 3.878562, 11.018, 22.05728, 79.138963]; % Serial runtime (seconds)

t_parallel = [ % Parallel runtimes for different thread counts
    0.028001, 0.089096, 1.116179, 1.52396, 4.691029, 17.853263, 96.326148; % n = 2
    0.033892, 0.139956, 0.630356, 2.177106, 6.616843, 20.351811, 80.154131; % n = 4
    0.026488, 0.115834, 0.576149, 3.318689, 9.754526, 23.157662, 83.627556; % n = 8
    0.072855, 0.063143, 0.447809, 3.402987, 13.813126, 32.680899, 100.585141;  % n = 16
    0.038374, 0.111952, 0.578477, 3.052137, 6.48275, 32.226204, 187.262787 % n = 32
];

threads = [2, 4, 8, 16, 32]; % Thread counts
colors = lines(length(threads)); % Generate distinct colors for each thread count

figure;
hold on;

% Plot serial result in black
plot(N, t_serial, 'k', 'LineWidth', 2, 'DisplayName', 'Serial');

% Plot parallel results with different colors
for i = 1:length(threads)
    plot(N, t_parallel(i, :), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', sprintf('n = %d', threads(i)));
end

hold off;
legend('Location', 'NorthWest');
xlabel('Node Count (N)');
ylabel('Runtime (Seconds)');
title('Runtime vs. Node Count for Serial and Parallel Execution');
grid on;

% Customize axis ticks to match power-of-2 spacing
set(gca, 'XTick', 2.^(11:17)); % Set X-ticks to powers of 2 from 2^0 to 2^18
set(gca, 'XTickLabel', {'2^11', '2^12', '2^13', '2^14', '2^15', '2^16', '2^17'}); % Set X-tick labels

% Uncommend the line below for better visualization 
% set(gca, 'XScale', 'log'); % Set X-axis scale to logarithmic if needed for better visualization
%}