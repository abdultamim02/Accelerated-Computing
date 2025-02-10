N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]; % Node counts
t_serial = [0.000035, 0.000045, 0.000056, 0.000185, 0.000424, 0.00366, 0.000526, 0.003601, 0.010152, 0.012711, 0.023983, 0.101417, 0.077975, 0.323897, 1.622221, 5.394472, 27.020556, 133.983921, 629.864252]; % Serial runtime (seconds)

t_parallel = [ % Parallel runtimes for different thread counts
    0.012209, 0.004637, 0.021334, 0.004419, 0.004676, 0.013269, 0.029884, 0.03459, 0.020792, 0.087327, 0.017143, 0.150361, 0.279482, 0.453569, 1.582676, 7.019541, 37.828138, 196.074999, 939.100715; % n = 2
    0.009197, 0.019862, 0.014287, 0.044311, 0.036312, 0.048621, 0.050932, 0.04461, 0.015046, 0.227271, 0.074767, 0.226012, 0.304112, 1.264248, 3.445512, 9.218669, 37.13049, 157.321492, 908.415491; % n = 4
    0.031926, 0.017469, 0.010416, 0.005059, 0.047273, 0.087025, 0.005984, 0.053017, 0.072723, 0.024241, 0.184077, 0.372035, 0.306402, 0.625733, 2.448076, 9.082046, 38.771025, 168.427069, 855.64699; % n = 8
    0.014489, 0.023013, 0.030505, 0.044463, 0.035913, 0.057487, 0.085352, 0.075711, 0.080681, 0.079652, 0.111239, 0.316205, 0.224595, 0.825756, 3.55949, 11.148855, 47.869237, 166.921253, 859.197172;  % n = 16
    0.041997, 0.068822, 0.045646, 0.044107, 0.049763, 0.039567, 0.087774, 0.094624, 0.133077, 0.181669, 0.337089, 0.301175, 0.317053, 0.627462, 3.623589, 12.147075, 37.926755, 152.680287, 783.424008 % n = 32
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