function plot_capability_map_matlab(jsonPath, outDir, scale)
% Generate only the 3 capability maps in MATLAB, with draggable model labels.
%
% Usage:
%   plot_capability_map_matlab('video-cloze/eval_results/analyze_report_all.json', 'figure/pics/capability_maps', 1.0)
%
% Notes:
% - Follows current Python logic:
%   1) filter by acc outside [0.23, 0.27] and acc~=0
%   2) top13 by overall acc, keep ~half (1st,3rd,5th,...) as green triangles
%   3) non-top13 shown as orange circles
% - Labels are draggable by mouse in each figure window.

if nargin < 1 || isempty(jsonPath)
    jsonPath = '/users/henry/Temporal-Cloze/video-cloze/eval_results/analyze_report_all.json';
end
if nargin < 2 || isempty(outDir)
    outDir = '/users/henry/Temporal-Cloze/figure/pics/capability_maps';
end
if nargin < 3 || isempty(scale)
    scale = 1.0;
end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

figPos = [120, 80, 920, 800];
% Keep canvas size fixed; scale only controls internal elements.
markerSize = 340 * (scale ^ 2);
markerLineWidth = 2.0 * scale;
refLineWidth = 2.2 * scale;
diagLineWidth = 2.4 * scale;
modelFontSize = 28 * scale;
eqFontSize = 28 * scale;
axisFontSize = 32 * scale;
labelFontSize = 36 * scale;
legendFontSize = 32 * scale;
legendMarkerSize = 16 * scale;
dxBase = 0.0025 * scale;
dyBase = 0.0015 * scale;
eqOffset = min(0.08, 0.02 * scale);

raw = fileread(jsonPath);
data = jsondecode(raw);
modelNameMap = build_model_name_map(raw);

rows = build_rows(data, modelNameMap);
if isempty(rows)
    error('No valid model rows found in %s', jsonPath);
end

% Match Python default filtering in main()
acc = [rows.acc];
keep = ((acc < 0.23) | (acc > 0.27)) & (acc ~= 0);
rows = rows(keep);
if isempty(rows)
    error('No models left after filtering by overall acc.');
end

% Sort by overall accuracy (desc)
[~, idx] = sort([rows.acc], 'descend');
rows = rows(idx);

topK = min(13, numel(rows));
topRows = rows(1:topK);
keepIdxInTop = 1:2:topK;  % keep 1st, 3rd, 5th... (about half)
keepTopRows = topRows(keepIdxInTop);

topNames = string({topRows.model});
keepTopNames = string({keepTopRows.model});
allNames = string({rows.model});

isTop = ismember(allNames, topNames);
isKeepTop = ismember(allNames, keepTopNames);

restRows = rows(~isTop);
plotTopRows = rows(isKeepTop);
plotRows = [restRows, plotTopRows];

pairs = {
    'A_acc', 'P_acc', 'A Accuracy', 'P Accuracy', 'A vs P', 'figure_02_capability_map_A_vs_P_v2';
    'P_acc', 'S_acc', 'P Accuracy', 'S Accuracy', 'P vs S', 'figure_02_capability_map_P_vs_S_v2';
    'S_acc', 'A_acc', 'S Accuracy', 'A Accuracy', 'S vs A', 'figure_02_capability_map_S_vs_A_v2';
};

for i = 1:size(pairs, 1)
    xField = pairs{i, 1};
    yField = pairs{i, 2};
    xLab = pairs{i, 3};
    yLab = pairs{i, 4};
    ttl = pairs{i, 5};
    stem = pairs{i, 6};

    fig = figure('Color', 'w', 'Position', figPos);
    ax = axes(fig);
    hold(ax, 'on');

    % Open-Source (orange circles)
    if ~isempty(restRows)
        xr = [restRows.(xField)];
        yr = [restRows.(yField)];
        scatter(ax, xr, yr, markerSize, ...
            'Marker', 'o', ...
            'MarkerFaceColor', [245, 133, 24] / 255, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', markerLineWidth, ...
            'DisplayName', 'Open-Source');
    end

    % Proprietary (green triangles)
    if ~isempty(plotTopRows)
        xt = [plotTopRows.(xField)];
        yt = [plotTopRows.(yField)];
        scatter(ax, xt, yt, markerSize, ...
            'Marker', '^', ...
            'MarkerFaceColor', [84, 162, 75] / 255, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', markerLineWidth, ...
            'DisplayName', 'Proprietary');
    end

    % Labels with small staggered offsets
    txtHandles = gobjects(1, numel(plotRows));
    for k = 1:numel(plotRows)
        r = plotRows(k);
        if mod(k, 2) == 0
            dx = -dxBase;
        else
            dx = dxBase;
        end
        if mod(floor((k - 1) / 2), 2) == 0
            dy = dyBase;
        else
            dy = -dyBase;
        end
        txtHandles(k) = text(ax, r.(xField) + dx, r.(yField) + dy, r.model, ...
            'FontSize', modelFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
    end

    % Reference lines
    xline(ax, 0.25, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', refLineWidth, 'HandleVisibility', 'off');
    yline(ax, 0.25, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', refLineWidth, 'HandleVisibility', 'off');

    xLower = min([plotRows.(xField)]) - 0.1;
    yLower = min([plotRows.(yField)]) - 0.1;
    xUpper = 1.0;
    yUpper = 1.0;

    xx = [max(xLower, yLower), min(xUpper, yUpper)];
    plot(ax, xx, xx, '--', 'Color', [0.62, 0.62, 0.62], 'LineWidth', diagLineWidth, 'HandleVisibility', 'off');

    xVar = upper(extractBefore(xField, '_'));
    yVar = upper(extractBefore(yField, '_'));
    eqLabel = sprintf('%s=%s', yVar, xVar);
    eqHandle = text(ax, xx(2) - eqOffset, xx(2) - eqOffset, eqLabel, ...
        'FontSize', eqFontSize, 'FontWeight', 'bold', 'Color', [0.45, 0.45, 0.45], ...
        'Rotation', 45, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
        'Interpreter', 'none');

    xlim(ax, [xLower, xUpper]);
    ylim(ax, [yLower, yUpper]);

    xTickStart = ceil(xLower / 0.3) * 0.3;
    yTickStart = ceil(yLower / 0.3) * 0.3;
    xticks(ax, xTickStart:0.3:xUpper);
    yticks(ax, yTickStart:0.3:yUpper);

    grid(ax, 'on');
    ax.GridAlpha = 0.22;
    ax.FontSize = axisFontSize;
    ax.FontWeight = 'bold';

    xlabel(ax, xLab, 'FontSize', labelFontSize, 'FontWeight', 'bold');
    ylabel(ax, yLab, 'FontSize', labelFontSize, 'FontWeight', 'bold');

    legHandles = gobjects(0);
    legLabels = {};
    if ~isempty(restRows)
        hLegOpen = plot(ax, nan, nan, 'o', ...
            'MarkerFaceColor', [245, 133, 24] / 255, ...
            'MarkerEdgeColor', 'k', ...
            'LineStyle', 'none', ...
            'LineWidth', markerLineWidth, ...
            'MarkerSize', legendMarkerSize);
        legHandles(end + 1) = hLegOpen; %#ok<AGROW>
        legLabels{end + 1} = 'Open-Source'; %#ok<AGROW>
    end
    if ~isempty(plotTopRows)
        hLegProp = plot(ax, nan, nan, '^', ...
            'MarkerFaceColor', [84, 162, 75] / 255, ...
            'MarkerEdgeColor', 'k', ...
            'LineStyle', 'none', ...
            'LineWidth', markerLineWidth, ...
            'MarkerSize', legendMarkerSize);
        legHandles(end + 1) = hLegProp; %#ok<AGROW>
        legLabels{end + 1} = 'Proprietary'; %#ok<AGROW>
    end
    if ~isempty(legHandles)
        legend(ax, legHandles, legLabels, ...
            'Location', 'southeast', ...
            'FontSize', legendFontSize, ...
            'FontWeight', 'bold', ...
            'Box', 'off');
    end

    txtHandles(end + 1) = eqHandle;

    enable_text_drag(fig, txtHandles);

    % Save both editable .fig and .pdf
    savefig(fig, fullfile(outDir, [stem '.fig']));
    exportgraphics(fig, fullfile(outDir, [stem '.pdf']), 'ContentType', 'vector');
end

fprintf('[OK] Wrote 3 capability maps to: %s (scale=%.2f)', outDir, scale);
end


function rows = build_rows(data, modelNameMap)
models = fieldnames(data.models);
rows = struct('model', {}, 'S_acc', {}, 'A_acc', {}, 'C_acc', {}, 'acc', {});

for i = 1:numel(models)
    name = models{i};
    m = data.models.(name);

    if ~isfield(m, 'S_acc') || ~isfield(m, 'A_acc') || ~isfield(m, 'C_acc') || ~isfield(m, 'acc')
        continue;
    end

    s = double(m.S_acc);
    a = double(m.A_acc);
    c = double(m.C_acc);
    o = double(m.acc);

    if any(~isfinite([s, a, c, o]))
        continue;
    end

    displayName = name;
    if nargin >= 2 && ~isempty(modelNameMap) && isKey(modelNameMap, name)
        displayName = modelNameMap(name);
    end

    row.model = displayName;
    row.S_acc = s;
    row.A_acc = a;
    row.C_acc = c;
    row.acc = o;
    rows(end + 1) = row; %#ok<AGROW>
end
end


function modelNameMap = build_model_name_map(raw)
% Recover original JSON model keys (preserve punctuation like . and -).
modelNameMap = containers.Map('KeyType', 'char', 'ValueType', 'char');

pattern = '"([^"\\]+)"\s*:\s*\{\s*"num_stems"\s*:';
tokens = regexp(raw, pattern, 'tokens');
for i = 1:numel(tokens)
    orig = tokens{i}{1};
    key = matlab.lang.makeValidName(orig, 'ReplacementStyle', 'underscore');
    if ~isKey(modelNameMap, key)
        modelNameMap(key) = orig;
    end
end
end


function enable_text_drag(fig, txtHandles)
% Click and drag text labels in figure.
setappdata(fig, 'drag_text_handles', txtHandles);
setappdata(fig, 'drag_text_current', []);

fig.WindowButtonDownFcn = @on_mouse_down;
fig.WindowButtonUpFcn = @on_mouse_up;

    function on_mouse_down(src, ~)
        h = hittest(src);
        if isa(h, 'matlab.graphics.primitive.Text')
            txt = getappdata(src, 'drag_text_handles');
            if any(h == txt)
                setappdata(src, 'drag_text_current', h);
                src.WindowButtonMotionFcn = @on_mouse_move;
            end
        end
    end

    function on_mouse_move(src, ~)
        h = getappdata(src, 'drag_text_current');
        if isempty(h) || ~isvalid(h)
            return;
        end
        ax = ancestor(h, 'axes');
        cp = ax.CurrentPoint;
        pos = h.Position;
        pos(1) = cp(1, 1);
        pos(2) = cp(1, 2);
        h.Position = pos;
    end

    function on_mouse_up(src, ~)
        src.WindowButtonMotionFcn = '';
        setappdata(src, 'drag_text_current', []);
    end
end
