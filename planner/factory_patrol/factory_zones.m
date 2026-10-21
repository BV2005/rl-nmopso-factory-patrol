function Z = factory_zones(L)
% Balanced power (weighted Voronoi) diagram zoning, after patrol_drone firmware/core/pd_zone.c
% (docs/algorithms.md section 9): world point p belongs to zone d iff d minimises
% |p - seed(d)|^2 - w(d). Two changes for the NMOPSO demo with a shared launch area:
%   - the generators are not the drones' homes (all homes sit together on the launch pads) but
%     seeds spread over the factory: k-means (Lloyd) centroids of the patrol lane network;
%   - the weights are balanced on the patrol lane length inside each zone rather than on
%     free-cell area (the storage aisles are 2.5 m apart, the other lanes up to 5.6 m, so equal
%     area would not mean equal flying).
% Weight update as in pd_zone_build(): w(d) -= k * (load(d) - mean(load)).
% Zones are assigned to drones in west-to-east order of their seeds (pads are west to east too).
% Returns Z.seeds, Z.w, Z.load (lane length per zone), Z.polys{d} (zone polygon, hall clipped),
% Z.pieces{d} (lane pieces [x1 y1 x2 y2] at least L.zoneMargin inside the zone), Z.walls{d}
% (cylinders [x y R] lining the zone's inner boundaries, for NMOPSO).
n = size(L.homes, 1);
ds = 0.1;                                              % lane sampling step (m)
P = zeros(0, 2); laneOf = zeros(0, 1);
for i = 1:size(L.laneNet, 1)
    s = L.laneNet(i, :); len = hypot(s(3) - s(1), s(4) - s(2));
    t = linspace(0, 1, max(2, round(len / ds) + 1))';
    P = [P; s(1) + t * (s(3) - s(1)), s(2) + t * (s(4) - s(2))]; %#ok<AGROW>
    laneOf = [laneOf; repmat(i, numel(t), 1)]; %#ok<AGROW>
end

% seeds: k-means on the lane samples, deterministic farthest-point initialisation
D2 = @(A, B) sum(A.^2, 2) - 2 * A * B' + sum(B.^2, 2)';      % squared distances, rows x rows
c = P(find(P(:, 1) == min(P(:, 1)), 1), :);
while size(c, 1) < n
    [~, j] = max(min(D2(P, c), [], 2)); c = [c; P(j, :)]; %#ok<AGROW>
end
for it = 1:200
    [~, a] = min(D2(P, c), [], 2);
    cn = c;
    for d = 1:n, cn(d, :) = mean(P(a == d, :), 1); end
    if max(abs(cn(:) - c(:))) < 1e-6, break, end
    c = cn;
end
[~, o] = sort(c(:, 1)); Z.seeds = c(o, :);
if isfield(L, 'assignZones') && strcmp(L.assignZones, 'optimal')
    % zone -> drone by optimal assignment: the permutation of seeds with the smallest total distance
    % from each drone's start to its seed (all n! permutations; n = 3 here)
    pm = perms(1:n); cost = zeros(size(pm, 1), 1);
    for k = 1:size(pm, 1)
        cost(k) = sum(vecnorm(L.homes - Z.seeds(pm(k, :), :), 2, 2));
    end
    [~, k] = min(cost); Z.seeds = Z.seeds(pm(k, :), :);
end

if isfield(L, 'balanceOn') && strcmp(L.balanceOn, 'tour')
    % balance the weights on each drone's estimated flight: the closed tour over its lane pieces
    % (tour_order.m, straight-line links) plus the transit from its start to the tour and back.
    % The load jumps when a piece changes zone, so the step shrinks and the best split is kept.
    w = zeros(n, 1); k = 0.5; best = inf;
    for it = 1:600
        a = owner(Z.seeds, w, P); load = zeros(n, 1);
        for d = 1:n
            pc = piecesOf(P, laneOf, a == d);
            [~, ~, tl] = tour_order(pc, L.homes(d, :));
            tr = 0; if ~isempty(pc), tr = 2 * min(vecnorm([pc(:, 1:2); pc(:, 3:4)] - L.homes(d, :), 2, 2)); end
            load(d) = tl + tr;
        end
        dev = max(abs(load - mean(load)));
        if dev < best, best = dev; bw = w; bload = load; end
        if dev < 2.0, break, end
        w = w - k * (load - mean(load)); k = k * 0.995;
    end
    w = bw; load = bload;
else
    % balance the weights on lane length (power-diagram assignment of the lane samples)
    w = zeros(n, 1); k = 0.5;
    for it = 1:2000
        load = ds * accumarray(owner(Z.seeds, w, P), 1, [n 1]);
        if max(abs(load - mean(load))) < 1.0, break, end
        w = w - k * (load - mean(load));
    end
end
Z.w = w; Z.load = load; Z.balanceIters = it;

% zone polygons: the hall rectangle clipped by every bisector half-plane
H = L.hall;
for d = 1:n
    poly = [H(1) H(3); H(2) H(3); H(2) H(4); H(1) H(4)];
    for e = [1:d-1, d+1:n]
        [nv, b] = bisector(Z, d, e);                   % p in d's half iff nv*p' + b <= 0
        poly = clipHalfPlane(poly, nv, b);
    end
    Z.polys{d} = poly;
end

% lane pieces: runs of lane samples at least L.zoneMargin inside the zone
for d = 1:n
    pc = zeros(0, 4);
    for i = 1:size(L.laneNet, 1)
        s = L.laneNet(i, :); len = hypot(s(3) - s(1), s(4) - s(2));
        t = linspace(0, 1, max(2, round(len / ds) + 1))';
        q = [s(1) + t * (s(3) - s(1)), s(2) + t * (s(4) - s(2))];
        in = arrayfun(@(j) zone_contains(Z, d, q(j, :), L.zoneMargin), (1:size(q, 1))');
        r = diff([0; in; 0]); st = find(r == 1); en = find(r == -1) - 1;
        for j = 1:numel(st)
            if norm(q(en(j), :) - q(st(j), :)) >= 1.0, pc = [pc; q(st(j), :), q(en(j), :)]; end %#ok<AGROW>
        end
    end
    Z.pieces{d} = pc;
end

% walls: cylinders of radius L.wallR just outside every inner edge of the zone polygon (surface
% on the boundary), L.wallSpacing apart, so NMOPSO's paths stay inside the zone
for d = 1:n
    poly = Z.polys{d}; cen = mean(poly, 1); wl = zeros(0, 3);
    for j = 1:size(poly, 1)
        a = poly(j, :); b = poly(mod(j, size(poly, 1)) + 1, :);
        onHall = (abs(a(1) - b(1)) < 1e-9 && any(abs(a(1) - H(1:2)) < 1e-9)) || ...
                 (abs(a(2) - b(2)) < 1e-9 && any(abs(a(2) - H(3:4)) < 1e-9));
        if onHall, continue, end
        e = b - a; len = norm(e); nrm = [e(2), -e(1)] / len;
        if dot(nrm, (a + b) / 2 - cen) < 0, nrm = -nrm; end     % outward
        t = linspace(0, 1, max(2, ceil(len / L.wallSpacing) + 1))';
        wl = [wl; a + t * e + L.wallR * nrm, repmat(L.wallR, numel(t), 1)]; %#ok<AGROW>
    end
    Z.walls{d} = wl;
end
end

function pc = piecesOf(P, laneOf, in)
% lane pieces [x1 y1 x2 y2] formed by the runs of lane samples marked `in` (at least 1 m long)
pc = zeros(0, 4);
for i = unique(laneOf)'
    idx = find(laneOf == i); m = in(idx);
    r = diff([0; m; 0]); st = find(r == 1); en = find(r == -1) - 1;
    for j = 1:numel(st)
        a = P(idx(st(j)), :); b = P(idx(en(j)), :);
        if norm(b - a) >= 1.0, pc = [pc; a b]; end %#ok<AGROW>
    end
end
end

function a = owner(seeds, w, P)
[~, a] = min(sum(P.^2, 2) - 2 * P * seeds' + sum(seeds.^2, 2)' - w', [], 2);
end

function [nv, b] = bisector(Z, d, e)
% p is in d's half-plane against e iff nv * p' + b <= 0 (docs/algorithms.md: g(p) <= 0)
hd = Z.seeds(d, :); he = Z.seeds(e, :);
nv = 2 * (he - hd);
b = sum(hd.^2) - sum(he.^2) - Z.w(d) + Z.w(e);
end

function out = clipHalfPlane(poly, nv, b)
% Sutherland-Hodgman: keep the part of the convex polygon with nv * p' + b <= 0
out = zeros(0, 2); m = size(poly, 1);
for j = 1:m
    p = poly(j, :); q = poly(mod(j, m) + 1, :);
    fp = nv * p' + b; fq = nv * q' + b;
    if fp <= 0, out = [out; p]; end %#ok<AGROW>
    if fp * fq < 0, out = [out; p + fp / (fp - fq) * (q - p)]; end %#ok<AGROW>
end
end
