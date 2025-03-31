close all
clear all
clc

% List of MATLAB files to process
file_list = {'T100.m', 'T1000.m', 'T10000.m', 'T100000.m', 'T1000000.m'}; 

for i = 1:length(file_list)
    filename = file_list{i};

    % Check if the file exists before running
    if exist(filename, 'file') == 2
        run(filename);  % Load T All Files

        % Check if T exists and is numeric
        if exist('T', 'var') == 1 && isnumeric(T)
            % Get matrix size and compute grid spacing dynamically
            [rows, cols] = size(T);
            n_cells = rows - 2;  % Since `T` includes boundary padding
            h = 1 / n_cells;      % Correct step size based on `n_cells`

            % Create matching meshgrid for T
            x = linspace(0, 1, cols);  % Ensure meshgrid matches `T`
            y = linspace(0, 1, rows);
            [X, Y] = meshgrid(x, y);

            % Create figure
            figure;
            surf(X, Y, T);
            set(gcf, 'color', 'w');
            title(['Iterations: ', filename(2:end-2)]); % Extract value of iterations
            xlabel('x');
            ylabel('y');
            zlim([min(T(:)) max(T(:))]); % Adjust Z-axis dynamically
            rotate3d on;
            shading interp;

            % Save the figure
            saveas(gcf, [filename(1:end-2), '.png']); % Saves as png image
        else
            warning(['Variable "T" not found in ', filename, '. Skipping...']);
        end
    else
        warning(['File ', filename, ' not found. Skipping...']);
    end
end