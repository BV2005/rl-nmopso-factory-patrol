function [lanes, info] = auto_lanes(L)
% Patrol lanes generated from the free space (no hand-placed lines): greedy coverage selection.
%   1. candidates: north-south and east-west lines every L.laneStep metres, cut into the pieces
%      that are free (free_space.m: clear of NMOPSO's obstacle cylinders by L.cylMargin, inside the
%      wall margin, away from the launch area), at least L.laneMinLen long;
%   2. what a piece sees: free floor cells within L.sensorReach of it, with a clear line of sight
%      (obstacles taller than the patrol band, e.g. the 3.5 m racks, block the view);
%   3. greedy: repeatedly take the piece that adds the most newly seen floor per metre flown
%      (length + L.laneTurnCost for getting to it), trimmed to the part that adds new floor,
%      until no piece adds at least L.laneMinGain m^2 per metre.
% In open floor this spaces lanes about 2 * sensorReach apart; between racks it finds the aisles.
% Returns lanes [x1 y1 x2 y2] and info (coverage of the floor that any lane could see, ...).
g = 0.25;                                               % coverage cell (m)
ds = 0.1;                                               % sampling along a candidate (m)
H = L.hall; m = L.wallMargin;
[gx, gy] = meshgrid(H(1) + g/2 : g : H(2), H(3) + g/2 : g : H(4));
free = true(size(gx));
for i = 1:size(L.rects, 1)
    r = L.rects(i, :);
    free = free & ~(gx >= r(1) & gx <= r(2) & gy >= r(3) & gy <= r(4));
end
tall = L.rects(L.rects(:, 5) > L.zmax, 1:4);            % block the view from the patrol band
keepout = [inf -inf inf -inf];                           % no launch area: nothing kept out
if isfield(L, 'launchArea'), keepout = L.launchArea + 1.5 * [-1 1 -1 1]; end   % no lanes over the pads
ok = @(x, y) free_space(L, x, y) & ~(x >= keepout(1) & x <= keepout(2) & y >= keepout(3) & y <= keepout(4));

% 1. candidate pieces [x1 y1 x2 y2]
C = zeros(0, 4);
for axisRun = 'yx'                                      % 'y': north-south lines, 'x': east-west
    if axisRun == 'y', across = H(1) + m : L.laneStep : H(2) - m; run = H(3) + m : ds : H(4) - m;
    else,              across = H(3) + m : L.laneStep : H(4) - m; run = H(1) + m : ds : H(2) - m; end
    for c = across
        if axisRun == 'y', f = ok(repmat(c, size(run)), run); else, f = ok(run, repmat(c, size(run))); end
        d = diff([0 f 0]); st = find(d == 1); en = find(d == -1) - 1;
        for k = 1:numel(st)
            if run(en(k)) - run(st(k)) < L.laneMinLen, continue, end
            if axisRun == 'y', C(end + 1, :) = [c run(st(k)) c run(en(k))]; %#ok<AGROW>
            else,              C(end + 1, :) = [run(st(k)) c run(en(k)) c]; end %#ok<AGROW>
        end
    end
end

% 2. floor seen by each candidate (linear cell indices), with line of sight, and where along the
%    candidate (metres from its start) each cell is seen from
seen = cell(size(C, 1), 1); foot = cell(size(C, 1), 1);
for i = 1:size(C, 1)
    [seen{i}, foot{i}] = sees(C(i, :), gx, gy, free, tall, L.sensorReach);
end
coverable = false(size(free)); coverable(vertcat(seen{:})) = true;

% 3. greedy selection: each candidate is scored on its best stretch [a, b] (the stretch with the
%    most newly seen floor per metre flown, at least laneMinLen long), not on its whole length
covered = false(size(free)); lanes = zeros(0, 4);
bw = 0.5;                                               % stretch resolution (m)
while true
    best = 0; bi = 0; bab = [0 0];
    for i = 1:size(C, 1)
        new = ~covered(seen{i});
        if ~any(new), continue, end
        len = norm(C(i, 3:4) - C(i, 1:2));
        nb = max(1, ceil(len / bw));
        cnt = accumarray(min(nb, floor(foot{i}(new) / bw) + 1), 1, [nb 1]);
        P = [0; cumsum(cnt)];
        [A, Bn] = ndgrid(0:nb - 1, 1:nb);                % stretch = bins A+1 .. Bn
        w = max((Bn - A) * bw, L.laneMinLen);
        sc = (P(Bn + 1) - P(A + 1)) * g^2 ./ (w + L.laneTurnCost);
        sc(Bn <= A) = 0;
        [v, k] = max(sc(:));
        if v > best, best = v; bi = i; bab = [A(k) * bw, min(len, Bn(k) * bw)]; end
    end
    if bi == 0 || best < L.laneMinGain, break, end
    s = stretch(C(bi, :), bab, L.laneMinLen);
    lanes(end + 1, :) = s; %#ok<AGROW>
    covered(sees(s, gx, gy, free, tall, L.sensorReach)) = true;
end
info.candidates = size(C, 1);
info.lanes = size(lanes, 1);
info.length = sum(vecnorm(lanes(:, 3:4) - lanes(:, 1:2), 2, 2));
info.coverOfCoverable = nnz(covered & coverable) / nnz(coverable);
info.coverOfFree = nnz(covered & free) / nnz(free);
end

function [idx, footM] = sees(s, gx, gy, free, tall, reach)
% free cells within `reach` of segment s whose view from the nearest point of s is not blocked;
% footM = where along s (metres from its start) each cell is seen from
a = s(1:2); e = s(3:4) - a;
near = gx >= min(s([1 3])) - reach & gx <= max(s([1 3])) + reach & ...
       gy >= min(s([2 4])) - reach & gy <= max(s([2 4])) + reach & free;
idx = find(near);
px = gx(idx); py = gy(idx);
t = max(0, min(1, ((px - a(1)) * e(1) + (py - a(2)) * e(2)) / max(e * e', eps)));
fx = a(1) + t * e(1); fy = a(2) + t * e(2);
keep = hypot(px - fx, py - fy) <= reach;
for q = linspace(0, 1, 9)                                % line of sight foot -> cell
    qx = fx + q * (px - fx); qy = fy + q * (py - fy);
    for i = 1:size(tall, 1)
        r = tall(i, :);
        keep = keep & ~(qx > r(1) & qx < r(2) & qy > r(3) & qy < r(4));
    end
end
idx = idx(keep);
footM = t(keep) * norm(e);
end

function s = stretch(s, ab, minLen)
% the part [ab(1), ab(2)] metres of candidate s, widened to at least minLen (within s)
a = s(1:2); e = s(3:4) - a; len = norm(e); u = e / max(len, eps);
if ab(2) - ab(1) < minLen
    mid = mean(ab); ab = [max(0, mid - minLen / 2), min(len, mid + minLen / 2)];
end
s = [a + ab(1) * u, a + ab(2) * u];
end
