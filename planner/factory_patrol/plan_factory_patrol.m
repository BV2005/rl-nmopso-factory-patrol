function plan_factory_patrol(nWorkers, authorsDir, timeLimitMin, layout, dryRun)
% Plans the 3-drone factory patrol with the authors' NMOPSO (unmodified NMOPSO.m).
%   - one zone per drone, with a closed lawnmower loop through it (see factory_layout)
%   - layout 'patrol_drone' (default): loop home -> lane points ... -> home, home inside the zone
%   - layout 'factory2' (shared start): all drones take off from one launch area; each flies a
%     transit leg to its zone, patrols a closed loop inside the zone, and flies a transit leg back
%   - every leg between consecutive patrol points is planned by NMOPSO.m (authorsDir, default
%     authors = the unmodified repository, MaxIt 500)
%   - per leg, the Pareto solution with the lowest threat cost F2 is flown (ties: lowest F1+F3+F4)
% Robustness: NMOPSO.m re-samples its initial swarm until one path is feasible and never gives
% up, so (1) every leg is pre-checked and split at a detour point while its random initial paths
% are never feasible; (2) each leg runs as its own job with a time limit - a leg that exceeds it
% is cancelled, retried once with another seed, then split at a detour point; (3) each finished
% leg is saved to <out>/legs/ at once (a re-run skips it) and logged to <out>/leg_log.txt.
% Output (factory_patrol/out, or out_<layout> for other layouts): patrol_paths.json (world ENU
% metres, for the Gazebo demo), legs.mat, patrol_top.png, report.txt.
%   plan_factory_patrol(6, fullfile(root, 'authors'), 12)
%   plan_factory_patrol(6, [], 12, 'factory2')
% Layout 'factory2_voronoi': zones from a balanced power diagram (factory_zones.m, after
% patrol_drone's pd_zone.c); each loop visits the zone's lane pieces nearest-first, joined by
% routes on a clearance grid; every loop leg carries its zone ([x0 x1 y0 y1 zone] limits), and
% NMOPSO gets the zone's boundary cylinders near the leg so its path stays inside the zone.
% dryRun = true stops after the pre-check and writes preview_paths.json (patrol points only).
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
if nargin < 1 || isempty(nWorkers), nWorkers = 4; end
if nargin < 2 || isempty(authorsDir), authorsDir = fullfile(root, 'authors'); end
if nargin < 3 || isempty(timeLimitMin), timeLimitMin = 12; end
if nargin < 4 || isempty(layout), layout = 'patrol_drone'; end
if nargin < 5 || isempty(dryRun), dryRun = false; end
addpath(here, fullfile(root, 'src'));
L = factory_layout(layout);
if isfield(L, 'autoLanes')                             % lanes generated from the free space
    [L.laneNet, info] = auto_lanes(L);
    fprintf('auto lanes: %d lanes, %.1f m, from %d candidates; floor seen %.1f %% of what any lane could see, %.1f %% of the free floor\n', ...
            info.lanes, info.length, info.candidates, 100 * info.coverOfCoverable, 100 * info.coverOfFree);
end
if isfield(L, 'voronoi')
    L.Z = factory_zones(L);
    for d = 1:3
        p = L.Z.polys{d};
        L.zones(d, :) = [min(p(:, 1)) max(p(:, 1)) min(p(:, 2)) max(p(:, 2))];
    end
end
if strcmp(layout, 'patrol_drone'), outDir = fullfile(here, 'out'); else, outDir = fullfile(here, ['out_' layout]); end
if ~isfolder(outDir), mkdir(outDir); end
legDir = fullfile(outDir, 'legs'); if ~isfolder(legDir), mkdir(legDir); end
logFile = fullfile(outDir, 'leg_log.txt');
logf = @(varargin) appendLog(logFile, varargin{:});

%% patrol points, pre-checked and split where NMOPSO could never start
addpath(fullfile(here, 'model'), authorsDir);          % CreateModel (factory) before authors'
segs = struct('drone', {}, 'kind', {}, 'pts', {}, 'box', {});   % box = [x0 x1 y0 y1 (zone)] the legs stay in
for d = 1:3
    if isfield(L, 'directTransit')
        % take off to the patrol band at the start, NMOPSO transit to the zone's tour start through
        % the whole hall (no layer, no corridor), patrol, and back; conflicts between drones are
        % resolved later by departure delays from a timed conflict check
        home = [L.homes(d, :), L.zPatrol];
        loop = zoneTour(L, d, home);
        vin = route(L, home, loop(1, :), 0);
        vout = route(L, loop(end, :), home, 0);
        segs(end + 1) = struct('drone', d, 'kind', 'in', 'pts', [home; vin; loop(1, :)], 'box', [L.hall 0]); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'loop', 'pts', loop, 'box', [L.zones(d, :) d]); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'out', 'pts', [loop(end, :); vout; home], 'box', [L.hall 0]); %#ok<AGROW>
        continue
    end
    if isfield(L, 'voronoi') && zone_contains(L.Z, d, L.homes(d, :), 0)
        % the pad lies in the drone's own zone: take off to the patrol band and reach the lanes
        % inside the zone (no transit layer, so the other drones' transits pass >= 0.9 m above it)
        zP = L.zPatrol;
        home = [L.homes(d, :), zP];
        corr = [L.homes(d, 1), L.corridorY, zP];
        loop = zoneTour(L, d, corr);
        vin = route(L, corr, loop(1, :), d);
        vout = route(L, loop(end, :), corr, d);
        segs(end + 1) = struct('drone', d, 'kind', 'in', 'pts', [home; corr; vin; loop(1, :)], ...
                               'box', [L.zones(d, :) d]); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'loop', 'pts', loop, 'box', [L.zones(d, :) d]); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'out', 'pts', [loop(end, :); vout; corr; home], ...
                               'box', [L.zones(d, :) d]); %#ok<AGROW>
        continue
    end
    if isfield(L, 'voronoi')
        zT = L.zTransit;
        home = [L.homes(d, :), zT];                    % take-off climbs straight to the transit layer
        corr = [L.homes(d, 1), L.corridorY, zT];
        loop = zoneTour(L, d, corr);
        up = [loop(1, 1:2), zT];                       % above the zone entry point
        vin = route(L, corr, up, 0); vin(:, 3) = zT;
        vout = route(L, up, corr, 0); vout(:, 3) = zT;
        segs(end + 1) = struct('drone', d, 'kind', 'in', 'pts', [home; corr; vin; up; loop(1, :)], ...
                               'box', [L.hall 0]); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'loop', 'pts', loop, 'box', [L.zones(d, :) d]); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'out', 'pts', [loop(end, :); up; vout; corr; home], ...
                               'box', [L.hall 0]); %#ok<AGROW>
        continue
    end
    loop = lawnmower(L, d);
    if L.sharedStart
        home = [L.homes(d, :), L.zPatrol];
        corr = [L.homes(d, 1), L.corridorY, L.zPatrol];  % launch corridor exit above the own pad
        segs(end + 1) = struct('drone', d, 'kind', 'in', 'pts', subdivide(L, [home; corr; loop(1, :)]), 'box', L.hall); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'loop', 'pts', subdivide(L, loop), 'box', L.zones(d, :)); %#ok<AGROW>
        segs(end + 1) = struct('drone', d, 'kind', 'out', 'pts', subdivide(L, [loop(end, :); corr; home]), 'box', L.hall); %#ok<AGROW>
    else
        segs(end + 1) = struct('drone', d, 'kind', 'loop', 'pts', loop, 'box', L.zones(d, :)); %#ok<AGROW>
    end
end
if isfield(L, 'voronoi')                               % no NMOPSO legs shorter than 0.5 m
    for i = 1:numel(segs), segs(i).pts = mergeClose(segs(i).pts, 0.5); end
end
for i = 1:numel(segs)
    segs(i).pts = checkedTour(L, segs(i).pts, segs(i).box);
end
if dryRun                                              % patrol points only, no NMOPSO
    drones = struct([]);
    for d = 1:3
        pick = @(kind) segs([segs.drone] == d & strcmp({segs.kind}, kind));
        drones(end + 1).id = d; %#ok<AGROW>
        drones(end).zone = L.zones(d, :); drones(end).home = L.homes(d, :);
        s = pick('loop'); drones(end).patrol_points = s.pts; drones(end).waypoints = s.pts;
        s = pick('in'); drones(end).transit_in = s.pts;
        s = pick('out'); drones(end).transit_out = s.pts;
        if isfield(L, 'Z'), drones(end).zone_polygon = L.Z.polys{d}; end
    end
    rep = checkPatrol(L, drones);
    fprintf('%s\n', rep{:});
    exportJson(L, layout, drones, fullfile(outDir, 'preview_paths.json'));
    fprintf('dry run: %d legs; preview written to %s\n', sum(arrayfun(@(s) size(s.pts, 1) - 1, segs)), outDir);
    return
end
logf('start (%s): %d legs after the pre-check, %d workers, %g min limit per leg', layout, ...
     sum(arrayfun(@(s) size(s.pts, 1) - 1, segs)), nWorkers, timeLimitMin);

%% plan the legs as separate jobs (saved as they finish; slow legs retried or split)
paths = {fullfile(here, 'noplot'), fullfile(root, 'src', 'noplot'), fullfile(here, 'model'), here, authorsDir};
if isempty(gcp('nocreate'))
    c = parcluster('Processes'); c.NumWorkers = max(c.NumWorkers, nWorkers);
    parpool(c, nWorkers);
end
pool = gcp;
queue = pendingLegs(segs, legDir);                    % rows: {drone, a, b, attempt, box}
running = struct('F', {}, 'drone', {}, 'a', {}, 'b', {}, 'attempt', {}, 'box', {}, 't0', {});
while ~isempty(queue) || ~isempty(running)
    while numel(running) < nWorkers && ~isempty(queue)
        q = queue(1, :); queue(1, :) = [];
        seed = 7000 + 97 * q{4} + mod(sum(round(100 * [q{2} q{3}])), 5000);
        running(end + 1) = struct('F', parfeval(pool, @planLeg, 1, q{2}, q{3}, seed, paths, L, q{5}), ...
            'drone', q{1}, 'a', q{2}, 'b', q{3}, 'attempt', q{4}, 'box', q{5}, 't0', tic); %#ok<AGROW>
    end
    pause(2);
    k = 1;
    while k <= numel(running)
        R = running(k);
        el = toc(R.t0) / 60;
        if strcmp(R.F.State, 'finished')
            try
                r = fetchOutputs(R.F); %#ok<NASGU>
                save(legFile(legDir, R.drone, R.a, R.b), 'r');
                logf('drone %d  (%6.1f,%6.1f)->(%6.1f,%6.1f)  attempt %d  done in %5.1f min  F=%s', R.drone, ...
                     R.a(1:2), R.b(1:2), R.attempt, el, mat2str(r.F', 3));
            catch err
                logf('drone %d  (%6.1f,%6.1f)->(%6.1f,%6.1f)  attempt %d  FAILED: %s', R.drone, R.a(1:2), ...
                     R.b(1:2), R.attempt, err.message);
                queue(end + 1, :) = {R.drone, R.a, R.b, R.attempt + 1, R.box}; %#ok<AGROW>
            end
            running(k) = [];
        elseif el > timeLimitMin
            cancel(R.F);
            if R.attempt == 1
                logf('drone %d  (%6.1f,%6.1f)->(%6.1f,%6.1f)  timed out (%.0f min): retrying with another seed', ...
                     R.drone, R.a(1:2), R.b(1:2), el);
                queue(end + 1, :) = {R.drone, R.a, R.b, 2, R.box}; %#ok<AGROW>
            else
                mid = detourPoint(L, R.a, R.b, R.box);
                logf('drone %d  (%6.1f,%6.1f)->(%6.1f,%6.1f)  timed out again: split at (%.1f,%.1f)', ...
                     R.drone, R.a(1:2), R.b(1:2), mid(1:2));
                segs = insertPoint(segs, R.drone, R.a, R.b, mid);
                queue = [queue; {R.drone, R.a, mid, 1, R.box}; {R.drone, mid, R.b, 1, R.box}]; %#ok<AGROW>
            end
            running(k) = [];
        else
            k = k + 1;
        end
    end
end
logf('all legs planned');

%% collect the saved legs
legs = struct('drone', {}, 'kind', {}, 'k', {}, 'a', {}, 'b', {});
res = {};
for i = 1:numel(segs)
    P = segs(i).pts;
    for k = 1:size(P, 1) - 1
        if isVertical(P(k, :), P(k + 1, :))            % flown straight
            S.r = struct('path', zeros(0, 3), 'F', zeros(4, 1), 'time', 0, 'front', zeros(4, 0));
        else
            S = load(legFile(legDir, segs(i).drone, P(k, :), P(k + 1, :)));
        end
        legs(end + 1) = struct('drone', segs(i).drone, 'kind', segs(i).kind, 'k', k, 'a', P(k, :), 'b', P(k + 1, :)); %#ok<AGROW>
        res{end + 1} = S.r; %#ok<AGROW>
    end
end
save(fullfile(outDir, 'legs.mat'), 'legs', 'res', 'segs', 'L');

%% assemble loops, check, export
fprintf('\n%-6s %-5s %-4s %-26s %s\n', 'drone', 'part', 'leg', 'F = [F1 F2 F3 F4]', 'time');
for i = 1:numel(legs)
    fprintf('%-6d %-5s %-4d %-26s %.0f s\n', legs(i).drone, legs(i).kind, legs(i).k, mat2str(res{i}.F', 3), res{i}.time);
end
drones = struct([]);
for d = 1:3
    part = struct('in', zeros(0, 3), 'loop', zeros(0, 3), 'out', zeros(0, 3));
    cost = zeros(0, 4);
    for kind = {'in', 'loop', 'out'}
        sel = find([legs.drone] == d & strcmp({legs.kind}, kind{1}));
        if isempty(sel), continue; end
        wp = legs(sel(1)).a;
        for i = sel
            wp = [wp; res{i}.path; legs(i).b]; %#ok<AGROW>
            cost(end + 1, :) = res{i}.F'; %#ok<AGROW>
        end
        part.(kind{1}) = round(wp, 3);
    end
    s = segs([segs.drone] == d & strcmp({segs.kind}, 'loop'));
    drones(end + 1).id = d; %#ok<AGROW>
    drones(end).zone = L.zones(d, :);
    drones(end).home = L.homes(d, :);
    drones(end).altitude = L.zPatrol;                  % take-off / landing hover height
    if isfield(L, 'zTransit') && abs(segs(find([segs.drone] == d, 1)).pts(1, 3) - L.zTransit) < 1e-6
        drones(end).altitude = L.zTransit;             % climbs straight to the transit layer
    end
    drones(end).depart_delay = L.departDelay(d);
    drones(end).patrol_points = s.pts;
    drones(end).waypoints = part.loop;          % closed patrol loop (flown LAPS times)
    drones(end).transit_in = part.in;           % launch pad -> loop start (empty: loop starts at home)
    drones(end).transit_out = part.out;         % loop end -> launch pad
    drones(end).leg_costs = cost;
    if isfield(L, 'Z'), drones(end).zone_polygon = L.Z.polys{d}; end
end
rep = checkPatrol(L, drones);
fprintf('%s\n', rep{:});
fid = fopen(fullfile(outDir, 'report.txt'), 'w'); fprintf(fid, '%s\n', rep{:}); fclose(fid);
exportJson(L, layout, drones, fullfile(outDir, 'patrol_paths.json'));
plotPatrol(L, drones, fullfile(outDir, 'patrol_top.png'));
fprintf('Saved %s\n', outDir);
end

function exportJson(L, layout, drones, file)
J.layout = layout;
J.frame = 'world ENU metres, origin at hall centre (factory.sdf); z = height above floor';
J.planner = 'authors'' NMOPSO.m (unmodified), MaxIt from the authors copy used';
J.hall = L.hall;
J.obstacles = struct('name', L.names(:), 'rect', num2cell(L.rects(:, 1:4), 2));
J.overflown = struct('name', L.overflownNames(:), 'rect', num2cell(L.overflown(:, 1:4), 2), ...
                     'top', num2cell(L.overflown(:, 5)));   % low obstacles the patrol flies over
if isfield(L, 'world'), J.world = L.world; end
if isfield(L, 'laneNet'), J.lanes = L.laneNet; end
if isfield(L, 'Z')
    J.zoning = struct('method', 'balanced power diagram (patrol_drone pd_zone.c), balanced on lane length', ...
        'seeds', L.Z.seeds, 'weights', L.Z.w, 'lane_length', L.Z.load, 'balance_iterations', L.Z.balanceIters);
end
J.drones = drones;
fid = fopen(file, 'w');
fprintf(fid, '%s', jsonencode(J, 'PrettyPrint', true)); fclose(fid);
end

%% ------------------------------------------------------------------------------------------
function pts = lawnmower(L, d)
% lawnmower loop through zone d at patrol height. Home layout: home -> lane ends (alternating) ->
% home. Shared start: lane points only, closed by returning to the first one; the lanes are
% ordered so that the loop starts on the lane nearest the launch area
if isfield(L, 'lanes')
    pts = explicitLanes(L, d);
    return
end
z = L.zones(d, :);
m = L.laneSpacing / 2;
xa = max(z(1) + m, L.hall(1) + L.wallMargin + 0.5);
xb = min(z(2) - m, L.hall(2) - L.wallMargin - 0.5);
nl = ceil((xb - xa) / L.laneSpacing) + 1;
xs = linspace(xa, xb, nl);
yLo = max(L.hall(3) + 2, z(3) + 0.5); yHi = min(L.hall(4) - 2, z(4) - 0.5);
if L.sharedStart && abs(xs(end) - mean(L.homes(:, 1))) < abs(xs(1) - mean(L.homes(:, 1)))
    xs = fliplr(xs);
end
nSeg = ceil((yHi - yLo) / L.maxLeg);
if L.sharedStart, pts = zeros(0, 2); else, pts = L.homes(d, :); end
for j = 1:nl
    ys = linspace(yLo, yHi, nSeg + 1);
    if mod(j, 2) == 0, ys = fliplr(ys); end
    for q = 1:numel(ys)
        p = [xs(j) ys(q)];
        if q == 1 || q == numel(ys)
            pts = [pts; freePoint(L, p)]; %#ok<AGROW>   lane ends: moved inwards if blocked
        elseif isFree(L, p)
            pts = [pts; p]; %#ok<AGROW>                 interior points: dropped if blocked
        end
    end
end
if L.sharedStart, pts = [pts; pts(1, :)]; else, pts = [pts; L.homes(d, :)]; end
keep = [true; vecnorm(diff(pts), 2, 2) >= 1.0];     % drop points that (nearly) repeat the previous
pts = pts(keep, :);                                  % one, e.g. a lane end moved onto a lane point
pts(:, 3) = L.zPatrol;
end

function pts = explicitLanes(L, d)
% closed loop over zone d's explicit lanes (L.lanes{d}: axis = direction the lanes run along,
% at = their positions across it). The loop starts at the lane end nearest the launch area and
% alternates direction; interior points every <= L.maxLeg are kept where free. An even number of
% lanes closes along the start edge; an odd number closes along the far edge and back down the
% first lane, so the loop never cuts diagonally across racks or machines.
z = L.zones(d, :); ln = L.lanes{d};
c0 = mean(L.homes, 1);                                   % launch area centre
if ln.axis == 'y', run = 2; across = 1; else, run = 1; across = 2; end
lim = [max(L.hall(2*run - 1) + 2, z(2*run - 1) + 0.5), min(L.hall(2*run) - 2, z(2*run) - 0.5)];
at = sort(ln.at);
if abs(at(end) - c0(across)) < abs(at(1) - c0(across)), at = fliplr(at); end
ends = lim;
if abs(lim(2) - c0(run)) < abs(lim(1) - c0(run)), ends = fliplr(lim); end
s = linspace(ends(1), ends(2), ceil(abs(diff(lim)) / L.maxLeg) + 1);
P = zeros(0, 2);
for j = 1:numel(at)
    sj = s; if mod(j, 2) == 0, sj = fliplr(s); end
    P = [P; lanePoints(L, across, at(j), sj)]; %#ok<AGROW>
end
if mod(numel(at), 2) == 1                                % ended on the far edge
    P = [P; lanePoints(L, across, at(1), fliplr(s))];
else
    P = [P; P(1, :)];
end
for k = find(~arrayfun(@(k) isFree(L, P(k, :)), 1:size(P, 1)))
    warning('zone %d: lane end (%.2f, %.2f) is not free', d, P(k, :));
end
keep = [true; vecnorm(diff(P), 2, 2) >= 1.0];
pts = P(keep, :);
pts(:, 3) = L.zPatrol;
end

function P = lanePoints(L, across, c, s)
% points of one lane at position c (coordinate `across`) through running coordinates s: both ends,
% plus the interior points that are free
P = zeros(0, 2);
for q = 1:numel(s)
    p = zeros(1, 2); p(across) = c; p(3 - across) = s(q);
    if q == 1 || q == numel(s) || isFree(L, p), P = [P; p]; end %#ok<AGROW>
end
end

function out = subdivide(L, pts)
% adds free intermediate points on legs longer than L.maxLeg (blocked ones are left out; NMOPSO
% plans around the obstacle)
out = pts(1, :);
for k = 1:size(pts, 1) - 1
    a = pts(k, :); b = pts(k + 1, :);
    n = ceil(norm(b(1:2) - a(1:2)) / L.maxLeg);
    for t = (1:n - 1) / n
        q = a + t * (b - a);
        if isFree(L, q(1:2)), out = [out; q]; end %#ok<AGROW>
    end
    out = [out; b]; %#ok<AGROW>
end
end

function ok = isFree(L, q)
ok = free_space(L, q(1), q(2));
end

function p = freePoint(L, p)
% moves a patrol point along y towards the hall centre until it is 1 m clear of every inflated
% obstacle rectangle
for t = 0:0.25:10
    q = [p(1), p(2) - sign(p(2)) * t];
    if isFree(L, q), p = q; return, end
end
end

function dist = rectDist(R, q)
dx = max([R(:, 1) - q(1), zeros(size(R, 1), 1), q(1) - R(:, 2)], [], 2);
dy = max([R(:, 3) - q(2), zeros(size(R, 1), 1), q(2) - R(:, 4)], [], 2);
dist = hypot(dx, dy);
end

function pts = checkedTour(L, pts, box)
% splits every leg of the tour at detour points while NMOPSO.m could never start on it
out = pts(1, :);
for k = 1:size(pts, 1) - 1
    segs = splitLeg(L, pts(k, :), pts(k + 1, :), 0, box);
    out = [out; segs(:, 4:6)]; %#ok<AGROW>
end
pts = out;
end

function segs = splitLeg(L, a, b, depth, box)
% returns [a b] rows; splits the leg at a free detour point while random initial paths are
% never feasible (1000 samples), up to 3 levels deep
if isVertical(a, b), segs = [a b]; return, end         % climb / descent: flown straight, no NMOPSO
p = initFeasibility(L, a, b, 1000, box);
if p > 0 || depth >= 3
    if p == 0, warning('leg (%.1f,%.1f)->(%.1f,%.1f) still has no feasible initial path', a(1), a(2), b(1), b(2)); end
    segs = [a b];
    fprintf('   leg (%6.1f,%6.1f) -> (%6.1f,%6.1f): %.1f %% random paths feasible\n', a(1:2), b(1:2), 100 * p);
    return
end
mid = detourPoint(L, a, b, box);
if isempty(mid), segs = [a b]; return, end
segs = [splitLeg(L, a, mid, depth + 1, box); splitLeg(L, mid, b, depth + 1, box)];
end

function mid = detourPoint(L, a, b, box)
% free point near the middle of leg a->b, searched along the leg's normal, inside box (empty if none)
m = (a + b) / 2;
n = [-(b(2) - a(2)), b(1) - a(1)] / max(norm(b(1:2) - a(1:2)), eps);   % unit normal of the leg
best = []; bestC = -inf;
for off = [0, 1:0.5:8, -(1:0.5:8)]
    q = m(1:2) + off * n;
    if q(1) < max(L.hall(1) + L.wallMargin, box(1) + 0.5) || q(1) > min(L.hall(2) - L.wallMargin, box(2) - 0.5) || ...
       q(2) < max(L.hall(3) + L.wallMargin, box(3) + 0.5) || q(2) > min(L.hall(4) - L.wallMargin, box(4) - 0.5), continue, end
    if numel(box) > 4 && box(5) > 0 && ~zone_contains(L.Z, box(5), q, L.routeMargin), continue, end
    if isFree(L, q)
        c = min(rectDist(L.rects(:, 1:4), q)) - 0.05 * abs(off);        % clear, but close to the line
        if c > bestC, bestC = c; best = q; end
        if off == 0, break, end
    end
end
mid = [];
if ~isempty(best), mid = [best, a(3)]; end
end

function segs = insertPoint(segs, d, a, b, mid)
for i = find([segs.drone] == d)
    P = segs(i).pts;
    k = find(all(abs(P(1:end-1, :) - a) < 1e-9, 2) & all(abs(P(2:end, :) - b) < 1e-9, 2), 1);
    if ~isempty(k), segs(i).pts = [P(1:k, :); mid; P(k+1:end, :)]; return, end
end
end

function q = pendingLegs(segs, legDir)
% legs of all segments without a saved result: rows {drone, a, b, attempt, box}
q = cell(0, 5);
for i = 1:numel(segs)
    P = segs(i).pts;
    for k = 1:size(P, 1) - 1
        a = P(k, :); b = P(k + 1, :);
        if isVertical(a, b), continue, end
        if ~isfile(legFile(legDir, segs(i).drone, a, b)), q(end + 1, :) = {segs(i).drone, a, b, 1, segs(i).box}; end %#ok<AGROW>
    end
end
end

function P = mergeClose(P, dmin)
% drops points closer than dmin (3D) to the previous kept point; the first and last points stay
% (a kept point too close before the last one is dropped instead)
keep = true(size(P, 1), 1); last = 1;
for k = 2:size(P, 1) - 1
    if norm(P(k, :) - P(last, :)) < dmin, keep(k) = false; else, last = k; end
end
kept = find(keep);
if numel(kept) > 2 && norm(P(kept(end), :) - P(kept(end - 1), :)) < dmin, keep(kept(end - 1)) = false; end
P = P(keep, :);
end

function v = isVertical(a, b)
% a pure climb or descent (transit layer <-> patrol band, take-off / landing): not an NMOPSO leg
v = norm(a(1:2) - b(1:2)) < 0.05 && abs(a(3) - b(3)) > 0.05;
end

function f = legFile(legDir, d, a, b)
% cached result of leg a -> b; legs off the nominal 1.8 m patrol height (transit layer) carry
% their heights in the name too (older plans were all at 1.8 m and keep their names)
if abs(a(3) - 1.8) < 1e-6 && abs(b(3) - 1.8) < 1e-6
    f = fullfile(legDir, sprintf('d%d_%.2f_%.2f_%.2f_%.2f.mat', d, a(1), a(2), b(1), b(2)));
else
    f = fullfile(legDir, sprintf('d%d_%.2f_%.2f_%.2f_%.2f_z%.2f_%.2f.mat', d, a(1), a(2), b(1), b(2), a(3), b(3)));
end
end

function appendLog(file, varargin)
line = sprintf('%s  %s', datestr(now, 'HH:MM:SS'), sprintf(varargin{:})); %#ok<TNOW1,DATST>
fid = fopen(file, 'a'); fprintf(fid, '%s\n', line); fclose(fid);
fprintf('%s\n', line);
end

function f = saveLeg(L, a, b, box)
% leg file read by model/CreateModel.m
start_w = a; goal_w = b; layout = L.name; xlim_w = box(1:2); ylim_w = box(3:4); %#ok<NASGU>
extra_threats = zeros(0, 3); %#ok<NASGU>
zlim_w = [L.zmin L.zmax]; %#ok<NASGU>
if numel(box) > 4 && box(5) == 0 && isfield(L, 'zTransitBand')
    zlim_w = L.zTransitBand; %#ok<NASGU>              % transit leg: transit layer
end
if numel(box) > 4 && box(5) > 0                        % zone walls near the leg ([x y R] metres)
    W = L.Z.walls{box(5)};
    e = b(1:2) - a(1:2); t = max(0, min(1, ((W(:, 1:2) - a(1:2)) * e') / max(e * e', eps)));
    near = vecnorm(W(:, 1:2) - (a(1:2) + t * e), 2, 2) <= 1.5 * norm(e) + 2;
    extra_threats = W(near, :); %#ok<NASGU>
end
f = [tempname '.mat'];
save(f, 'start_w', 'goal_w', 'layout', 'xlim_w', 'ylim_w', 'zlim_w', 'extra_threats');
end

function pts = zoneTour(L, d, s0)
% closed patrol loop over zone d's lane pieces: start at the piece end nearest s0, fly each piece
% to its other end, go to the nearest end of a remaining piece (routed on the clearance grid inside
% the zone), and finally route back to the start
pc = L.Z.pieces{d};
E = [pc(:, 1:2); pc(:, 3:4)];                          % piece ends, row i and i + n are one piece
n = size(pc, 1);
if isfield(L, 'tourOrder') && strcmp(L.tourOrder, '2opt')
    % order and directions from tour_order.m (nearest neighbour + 2-opt on the closed cycle)
    [ord, flip] = tour_order(pc, s0);
    P = zeros(0, 2); cur = [];
    for k = 1:n
        i = ord(k);
        if flip(k), a = pc(i, 3:4); b = pc(i, 1:2); else, a = pc(i, 1:2); b = pc(i, 3:4); end
        if ~isempty(cur)
            R = route(L, [cur L.zPatrol], [a L.zPatrol], d); P = [P; R(:, 1:2)]; %#ok<AGROW>
        end
        m = ceil(norm(b - a) / L.maxLeg);
        P = [P; a + (0:m)' / m * (b - a)]; %#ok<AGROW>
        cur = b;
    end
    todo = false;                                      % skip the nearest-first loop below
else
    [~, j] = min(vecnorm(E - s0(1:2), 2, 2));
    todo = true(n, 1); P = zeros(0, 2); cur = [];
end
while any(todo)
    i = mod(j - 1, n) + 1;
    if j <= n, a = pc(i, 1:2); b = pc(i, 3:4); else, a = pc(i, 3:4); b = pc(i, 1:2); end
    if ~isempty(cur)
        R = route(L, [cur L.zPatrol], [a L.zPatrol], d); P = [P; R(:, 1:2)]; %#ok<AGROW>
    end
    m = ceil(norm(b - a) / L.maxLeg);
    P = [P; a + (0:m)' / m * (b - a)]; %#ok<AGROW>
    todo(i) = false; cur = b;
    if ~any(todo), break, end
    D = vecnorm(E - cur, 2, 2); D([~todo; ~todo]) = inf;
    [~, j] = min(D);
end
R = route(L, [cur L.zPatrol], [P(1, :) L.zPatrol], d);
P = [P; R(:, 1:2); P(1, :)];
% drop points closer than 1 m to the previous kept one; the closing point (= start) always stays,
% and a kept point too close before it is dropped instead
keep = true(size(P, 1), 1); last = 1;
for k = 2:size(P, 1) - 1
    if norm(P(k, :) - P(last, :)) < 1.0, keep(k) = false; else, last = k; end
end
kept = find(keep);
if numel(kept) > 2 && norm(P(kept(end), :) - P(kept(end - 1), :)) < 1.0, keep(kept(end - 1)) = false; end
pts = P(keep, :);
pts(:, 3) = L.zPatrol;
end

function V = route(L, a, b, zone)
% interior via points ([x y z]) of a clear route a -> b: shortest 8-connected path on a grid of
% free cells (obstacles inflated + L.freeMargin, wall margin, inside zone `zone` by
% L.routeMargin if zone > 0), shortened by line of sight, split into pieces <= L.maxLeg
g = L.gridRes;
xs = L.hall(1) + L.wallMargin : g : L.hall(2) - L.wallMargin;
ys = L.hall(3) + L.wallMargin : g : L.hall(4) - L.wallMargin;
[GX, GY] = meshgrid(xs, ys);
free = free_space(L, GX, GY);
if zone > 0
    for e = [1:zone-1, zone+1:numel(L.Z.w)]
        nv = 2 * (L.Z.seeds(e, :) - L.Z.seeds(zone, :));
        bb = sum(L.Z.seeds(zone, :).^2) - sum(L.Z.seeds(e, :).^2) - L.Z.w(zone) + L.Z.w(e);
        free = free & (nv(1) * GX + nv(2) * GY + bb) / norm(nv) <= -L.routeMargin;
    end
end
idx = find(free); [ny, nx] = size(free);
[~, s] = min((GX(idx) - a(1)).^2 + (GY(idx) - a(2)).^2);   % nearest free cells (graph node numbers)
[~, t] = min((GX(idx) - b(1)).^2 + (GY(idx) - b(2)).^2);
[r, c] = ind2sub([ny nx], idx);
lin = zeros(ny, nx); lin(idx) = 1:numel(idx);
S = []; T = []; Wt = [];
for off = [0 1; 1 0; 1 1; 1 -1]'
    r2 = r + off(1); c2 = c + off(2);
    ok = r2 >= 1 & r2 <= ny & c2 >= 1 & c2 <= nx;
    k2 = zeros(size(r)); k2(ok) = lin(sub2ind([ny nx], r2(ok), c2(ok)));
    ok = k2 > 0;
    S = [S; find(ok)]; T = [T; k2(ok)]; Wt = [Wt; repmat(g * norm(off), nnz(ok), 1)]; %#ok<AGROW>
end
G = graph(S, T, Wt, numel(idx));
path = shortestpath(G, s, t);
if isempty(path), error('route: no clear path (%.1f,%.1f) -> (%.1f,%.1f) in zone %d', a(1), a(2), b(1), b(2), zone); end
Q = [a(1:2); GX(idx(path)) GY(idx(path)); b(1:2)];
% line of sight: keep the farthest point reachable in a straight line over free cells
F = griddedInterpolant({ys, xs}, double(free), 'nearest');
clear_ = @(p, q) all(F(linspace(p(2), q(2), 50)', linspace(p(1), q(1), 50)') > 0.5);
K = 1; i = 1;
while i < size(Q, 1)
    j = size(Q, 1);
    while j > i + 1 && ~clear_(Q(i, :), Q(j, :)), j = j - 1; end
    K(end + 1) = j; i = j; %#ok<AGROW>
end
Q = Q(K, :);
V = zeros(0, 2);
for k = 1:size(Q, 1) - 1
    m = ceil(norm(Q(k + 1, :) - Q(k, :)) / L.maxLeg);
    V = [V; Q(k, :) + (1:m)' / m * (Q(k + 1, :) - Q(k, :))]; %#ok<AGROW>
end
V = V(1:end-1, :);                                     % interior points only (b is excluded)
V = V(vecnorm(V - a(1:2), 2, 2) >= 0.5 & vecnorm(V - b(1:2), 2, 2) >= 0.5, :);
V(:, 3) = L.zPatrol;
end

function p = initFeasibility(L, a, b, N, box)
% fraction of NMOPSO.m's random initial paths (its own bounds) that are feasible on leg a->b
f = saveLeg(L, a, b, box); setenv('NMOPSO_LEG_FILE', f);
model = CreateModel(); delete(f);
nVar = model.n; VarSize = [1 nVar];
VarMax.r = 3 * norm(model.start - model.end) / nVar; VarMin.r = VarMax.r / 9;
VarMin.psi = -pi/4; VarMax.psi = pi/4; VarMin.phi = -pi/4; VarMax.phi = pi/4;
ok = 0;
for i = 1:N
    c = MyCost(SphericalToCart2(CreateRandomSolution(VarSize, VarMin, VarMax), model), model, VarMin);
    ok = ok + all(isfinite(c));
end
p = ok / N;
end

function r = planLeg(a, b, seed, paths, L, box)
% runs the authors' NMOPSO.m on leg a->b and returns the chosen path (world metres)
for p = numel(paths):-1:1, addpath(paths{p}, '-begin'); end
f = saveLeg(L, a, b, box);
setenv('NMOPSO_LEG_FILE', f);
rng(seed, 'twister');
t0 = tic;
[rep, model] = runAuthorsScript();
r.time = toc(t0);
delete(f);
C = [rep.Cost];
ok = find(all(isfinite(C), 1));
C = C(:, ok);
f2 = C(2, :);
s = L.scale;
toWorld = @(P) [(P.x(:) - 1) / s + L.hall(1), (P.y(:) - 1) / s + L.hall(3), P.z(:) / s];
if isfield(L, 'select') && strcmp(L.select, 'short-safe')
    % shortest among the nearly safest: threat within 10 % of the front's threat range of the
    % minimum, then the lowest length cost F1 (ties: lowest F2, then F3 + F4)
    cand = find(f2 <= min(f2) + 0.1 * (max(f2) - min(f2)) + 1e-12);
    [~, o] = sortrows([C(1, cand)' C(2, cand)' sum(C([3 4], cand), 1)']);
    j = o(1);
else
    % safest: lowest threat F2 (ties: lowest F1 + F3 + F4)
    cand = find(f2 <= min(f2) + 1e-12);
    [~, j] = min(sum(C([1 3 4], cand), 1));
end
best = rep(ok(cand(j)));
r.path = toWorld(SphericalToCart2(best.Position, model));
r.F = C(:, cand(j));
r.front = C;
r.frontPaths = arrayfun(@(k) toWorld(SphericalToCart2(rep(k).Position, model)), ok, 'UniformOutput', false);
end

function [rep, model] = runAuthorsScript()
rep = []; model = [];            %#ok<NASGU> assigned by the script
evalc('NMOPSO');                 % authors' script, unmodified
close all force
end

function rep = checkPatrol(L, drones)
% coverage of the free floor (2 m sensor reach) by the patrol loops, and clearance of all flown
% polylines (transits included)
rep = {'', '== Patrol check'};
g = 0.25;
[gx, gy] = meshgrid(L.hall(1) + g/2:g:L.hall(2), L.hall(3) + g/2:g:L.hall(4));
free = true(size(gx));
for i = 1:size(L.rects, 1)
    r = L.rects(i, 1:4);
    free = free & ~(gx >= r(1) & gx <= r(2) & gy >= r(3) & gy <= r(4));
end
covered = false(size(gx));
for d = 1:numel(drones)
    W = drones(d).waypoints;
    Wall = [drones(d).transit_in; W; drones(d).transit_out];
    cz = false(size(gx));
    for k = 1:size(W, 1) - 1
        a = W(k, 1:2); b = W(k + 1, 1:2); e = b - a;
        t = max(0, min(1, ((gx - a(1)) * e(1) + (gy - a(2)) * e(2)) / max(e * e', eps)));
        cz = cz | hypot(a(1) + t * e(1) - gx, a(2) + t * e(2) - gy) <= 2.0;
    end
    minClr = inf;
    for k = 1:size(Wall, 1) - 1                    % clearance to the (uninflated) obstacles
        a = Wall(k, 1:2); e = Wall(k + 1, 1:2) - a;
        for q = linspace(0, 1, 40)
            minClr = min(minClr, min(rectDist(L.rects(:, 1:4), a + q * e)));
        end
    end
    z = L.zones(d, :);
    if isfield(L, 'Z')
        zoneMask = free & inpolygon(gx, gy, L.Z.polys{d}(:, 1), L.Z.polys{d}(:, 2));
    else
        zoneMask = free & gx >= z(1) & gx <= z(2) & gy >= z(3) & gy <= z(4);
    end
    covered = covered | cz;
    len = sum(vecnorm(diff(W), 2, 2));
    tl = sum(vecnorm(diff(drones(d).transit_in), 2, 2)) + sum(vecnorm(diff(drones(d).transit_out), 2, 2));
    rep{end + 1} = sprintf('drone %d: %d waypoints, loop length %.1f m (+ %.1f m transit), zone coverage %.1f %%, min clearance to obstacles %.2f m, altitude %.2f-%.2f m', ...
        drones(d).id, size(W, 1), len, tl, 100 * nnz(cz & zoneMask) / nnz(zoneMask), minClr, min(Wall(:, 3)), max(Wall(:, 3))); %#ok<AGROW>
end
rep{end + 1} = sprintf('whole factory: %.1f %% of the free floor within 2 m of a patrol loop', 100 * nnz(covered & free) / nnz(free));
end

function plotPatrol(L, drones, file)
f = figure('Visible', 'off', 'Position', [100 100 1100 800]);
hold on; axis equal; box on;
col = [42 120 214; 235 104 52; 27 175 122] / 255;             % validated categorical slots 1-3
for d = 1:3
    z = L.zones(d, :);
    if isfield(L, 'Z')
        fill(L.Z.polys{d}(:, 1), L.Z.polys{d}(:, 2), col(d, :), 'FaceAlpha', 0.08, 'EdgeColor', col(d, :), ...
             'HandleVisibility', 'off');
        plot(L.Z.seeds(d, 1), L.Z.seeds(d, 2), 'x', 'Color', col(d, :), 'MarkerSize', 12, 'LineWidth', 2, ...
             'HandleVisibility', 'off');
    else
        fill([z(1) z(2) z(2) z(1)], z([3 3 4 4]), col(d, :), 'FaceAlpha', 0.06, 'EdgeColor', 'none', ...
             'HandleVisibility', 'off');
    end
end
c = factory_cylinders(L);
th = linspace(0, 2 * pi, 60);
for i = 1:size(c, 1)
    plot(c(i, 1) + c(i, 3) * cos(th), c(i, 2) + c(i, 3) * sin(th), ':', 'Color', [0.6 0.6 0.6], ...
         'HandleVisibility', 'off');
end
for i = 1:size(L.rects, 1)
    r = L.rects(i, 1:4);
    fill(r([1 2 2 1]), r([3 3 4 4]), [0.45 0.45 0.45], 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
for i = 1:size(L.overflown, 1)                                 % low obstacles, flown over
    r = L.overflown(i, 1:4);
    fill(r([1 2 2 1]), r([3 3 4 4]), [0.82 0.82 0.82], 'EdgeColor', [0.6 0.6 0.6], 'LineStyle', ':', ...
         'HandleVisibility', 'off');
end
h = gobjects(1, numel(drones));
for d = 1:numel(drones)
    W = drones(d).waypoints; P = drones(d).patrol_points;
    h(d) = plot(W(:, 1), W(:, 2), '-', 'Color', col(d, :), 'LineWidth', 2);
    T = [drones(d).transit_in; nan(1, 3); drones(d).transit_out];
    if size(T, 1) > 1
        plot(T(:, 1), T(:, 2), '--', 'Color', col(d, :), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    plot(P(:, 1), P(:, 2), 'o', 'MarkerSize', 7, 'MarkerFaceColor', col(d, :), 'MarkerEdgeColor', 'w', ...
         'HandleVisibility', 'off');
    plot(drones(d).home(1), drones(d).home(2), 's', 'MarkerSize', 11, 'MarkerFaceColor', col(d, :), ...
         'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
rectangle('Position', [L.hall(1) L.hall(3) diff(L.hall(1:2)) diff(L.hall(3:4))], 'LineWidth', 1.5);
xlim(L.hall(1:2) + [-1 1]); ylim(L.hall(3:4) + [-1 1]);
xlabel('x [m]'); ylabel('y [m]');
legend(h, arrayfun(@(d) sprintf('Drone %d', d.id), drones, 'UniformOutput', false), 'Location', 'northeastoutside');
if L.sharedStart
    r = L.launchArea;
    plot(r([1 2 2 1 1]), r([3 3 4 4 3]), 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');
    sub = 'grey = obstacles (light: flown over); dotted = cylinders NMOPSO avoids; squares = launch pads; dashed = transit';
else
    sub = 'grey = obstacles (racks, machines, boxes); dotted = cylinders NMOPSO avoids; squares = homes';
end
title({'NMOPSO factory patrol: one zone and one closed loop per drone', sub}, 'FontWeight', 'normal');
try, theme(f, 'light'); catch, end
exportgraphics(f, file, 'Resolution', 150);
close(f);
end
