GlobalDevice = [0.261120, 0.345088, 0.346112, 0.279552, 0.267264, 0.272384, 0.266240, 0.259840, 0.274976, 1.022496, 7.343264, 62.927486, 456.953308, 4352.567383];
TiledDevice = [0.011264, 0.011264, 0.010240, 0.010240, 0.010240, 0.012288, 0.014336, 0.025600, 0.099328, 0.661504, 5.161984, 41.128960, 256.844788, 2050.566162];
mxn = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]; % Powers of 2

% Plot settings
figure;
hold on;
plot(mxn, GlobalDevice, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'GlobalDevice');
plot(mxn, TiledDevice, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'r', 'DisplayName', 'Tiled Device');
hold off;

% Formatting
xlabel('Matrix Size (m = n)');
ylabel('Elapsed Time (ms)');
title('Global Device vs. Tiled Device Performance');
grid on;

% Customize x-axis ticks and scale
set(gca, 'XScale', 'log');
set(gca, 'xtick', mxn);
set(gca, 'xticklabels', {'2^{1}', '2^{2}', '2^{3}', '2^{4}', '2^{5}', '2^{6}', '2^{7}', '2^{8}', '2^{9}', '2^{10}', '2^{11}', '2^{12}', '2^{13}', '2^{14}'});

% Legend
legend('Location', 'northwest');

% Save Plot
saveas(gcf, 'GlobalDevicevsTiledDevicePerformance.png');
