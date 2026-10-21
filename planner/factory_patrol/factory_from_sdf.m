function W = factory_from_sdf(file, flyOverTop)
% Reads the factory geometry from a Gazebo world file (e.g. Downloads/factory2), instead of typing it
% into factory_layout.m. Every static model with a box collision is classified by its geometry:
%   floor    - the largest thin box at ground level; its extent is the hall's outer extent
%   ceiling  - a large thin box well above the ground (skipped)
%   walls    - tall boxes lying along the floor's edges; their inner faces give the hall
%   markings - boxes at most 5 cm tall (floor lines, skipped)
%   others   - obstacles; those with top <= flyOverTop are low enough to fly over
% Model yaw is honoured by taking the axis-aligned bounding box of the rotated footprint.
% Returns W.hall [xmin xmax ymin ymax] (inner wall faces), W.roof, W.rects [xmin xmax ymin ymax top],
% W.names, W.overflown (same columns), W.overflownNames, W.skipped (names and reasons).
persistent cacheKey cacheW
d = dir(file);
key = sprintf('%s|%s|%g', file, d.date, flyOverTop);
if isequal(key, cacheKey), W = cacheW; return, end

doc = xmlread(file);
models = doc.getElementsByTagName('model');
B = zeros(0, 6); names = {};                      % boxes: [cx cy cz sx sy sz] (after yaw -> AABB)
for k = 0:models.getLength - 1
    m = models.item(k);
    name = char(m.getAttribute('name'));
    mp = poseOf(m);                               % model pose [x y z roll pitch yaw]
    cols = m.getElementsByTagName('collision');
    for c = 0:cols.getLength - 1
        col = cols.item(c);
        box = col.getElementsByTagName('box');
        if box.getLength == 0, continue, end
        sz = str2num(char(box.item(0).getElementsByTagName('size').item(0).getTextContent)); %#ok<ST2NM>
        cp = poseOf(col); lp = poseOf(col.getParentNode);   % collision and link offsets
        off = cp(1:3) + lp(1:3);
        yaw = mp(6) + lp(6) + cp(6);
        R = [cos(mp(6)) -sin(mp(6)); sin(mp(6)) cos(mp(6))];
        ctr = [mp(1:2) + (R * off(1:2)')', mp(3) + off(3)];
        ext = abs([cos(yaw) sin(yaw); sin(yaw) cos(yaw)]) * sz(1:2)';   % AABB of the rotated footprint
        B(end + 1, :) = [ctr, ext', sz(3)]; %#ok<AGROW>
        names{end + 1} = name; %#ok<AGROW>
    end
end
lo = B(:, 1:3) - B(:, 4:6) / 2; hi = B(:, 1:3) + B(:, 4:6) / 2;
area = B(:, 4) .* B(:, 5);
thin = B(:, 6) <= 0.2;

% floor: largest thin box touching the ground
cand = find(thin & hi(:, 3) <= 0.05 & lo(:, 3) >= -0.5);
[~, j] = max(area(cand)); f = cand(j);
outer = [lo(f, 1) hi(f, 1) lo(f, 2) hi(f, 2)];
skip = false(size(B, 1), 1); why = cell(size(B, 1), 1);
skip(f) = true; why{f} = 'floor';
% ceiling: large thin boxes high up
for i = find(thin & area >= 0.5 * area(f) & lo(:, 3) > 2)'
    skip(i) = true; why{i} = 'ceiling';
end
% walls: tall boxes along the floor edges, spanning most of an edge
hall = outer; roof = inf;
for i = find(~skip)'
    t = 0.5;                                      % how close to an edge counts as "on" it
    spanX = B(i, 4) >= 0.8 * (outer(2) - outer(1)); spanY = B(i, 5) >= 0.8 * (outer(4) - outer(3));
    if spanX && abs(B(i, 2) - outer(4)) < t, hall(4) = min(hall(4), lo(i, 2)); elseif ...
       spanX && abs(B(i, 2) - outer(3)) < t, hall(3) = max(hall(3), hi(i, 2)); elseif ...
       spanY && abs(B(i, 1) - outer(2)) < t, hall(2) = min(hall(2), lo(i, 1)); elseif ...
       spanY && abs(B(i, 1) - outer(1)) < t, hall(1) = max(hall(1), hi(i, 1)); else, continue
    end
    skip(i) = true; why{i} = 'wall';
    roof = min(roof, hi(i, 3));
end
for i = find(~skip & B(:, 6) <= 0.05)'
    skip(i) = true; why{i} = 'floor marking';
end
for i = find(skip & strcmp(why, 'ceiling'))', roof = min(roof, lo(i, 3)); end

keep = find(~skip);
rect = [lo(keep, 1) hi(keep, 1) lo(keep, 2) hi(keep, 2) hi(keep, 3)];
low = rect(:, 5) <= flyOverTop;
W.hall = hall; W.roof = roof;
W.rects = rect(~low, :); W.names = names(keep(~low));
W.overflown = rect(low, :); W.overflownNames = names(keep(low));
W.skipped = [names(skip); why(skip)']';
cacheKey = key; cacheW = W;
end

function p = poseOf(node)
% the node's own <pose> child (not a descendant's), [x y z roll pitch yaw]; zeros if absent
p = zeros(1, 6);
ch = node.getChildNodes;
for k = 0:ch.getLength - 1
    n = ch.item(k);
    if n.getNodeType == n.ELEMENT_NODE && strcmp(char(n.getNodeName), 'pose')
        v = str2num(char(n.getTextContent)); %#ok<ST2NM>
        p(1:numel(v)) = v;
        return
    end
end
end
