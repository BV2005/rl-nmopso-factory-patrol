function run_nmopso_legs(legFile, outDir, seeds, nWorkers)
% Runs the authors' unmodified NMOPSO.m (MaxIt 500, nPop 100, nRep 50) on every benchmark leg and
% seed, as parallel jobs (the pattern of plan_factory_patrol.m planLeg). One MAT per (leg, seed) in
% outDir, skipped when it exists, holding the whole final repository (Pareto front):
%   front [4 x m] costs, r/psi/phi [m x n] encodings, time (s).
% legFile (MAT): legs [K x 6] world metres [ax ay az bx by bz].
P = setup_paths();
if nargin < 3 || isempty(seeds), seeds = 1; end
if nargin < 4 || isempty(nWorkers), nWorkers = 6; end
I = load(legFile);
if ~isfolder(outDir), mkdir(outDir); end
if isempty(gcp('nocreate'))
    c = parcluster('Processes'); c.NumWorkers = max(c.NumWorkers, nWorkers); parpool(c, nWorkers);
end
pool = gcp;
L = factory_layout(P.layout);
F = parallel.FevalFuture.empty; names = {};
for k = 1:size(I.legs, 1)
    for s = seeds
        out = fullfile(outDir, sprintf('leg%03d_seed%d.mat', k, s));
        if isfile(out), continue, end
        F(end + 1) = parfeval(pool, @oneLeg, 1, I.legs(k, :), 1000 * s + k, P.paths, L); %#ok<AGROW>
        names{end + 1} = out; %#ok<AGROW>
    end
end
fprintf('%d NMOPSO jobs on %d workers\n', numel(F), pool.NumWorkers);
for j = 1:numel(F)
    [idx, r] = fetchNext(F);
    save(names{idx}, '-struct', 'r');
    fprintf('%s  %s  %.0f s, %d front points\n', datestr(now, 'HH:MM:SS'), names{idx}, r.time, size(r.front, 2)); %#ok<TNOW1,DATST>
end
end

function r = oneLeg(leg, seed, paths, L)
for p = numel(paths):-1:1, addpath(paths{p}, '-begin'); end
f = write_leg(L, leg(1:3), leg(4:6));
setenv('NMOPSO_LEG_FILE', f);
rng(seed, 'twister');
t0 = tic;
[rep, model] = runAuthorsScript();
r.time = toc(t0);
delete(f);
r.seed = seed; r.leg = leg;
r.front = [rep.Cost];
r.r = cell2mat(arrayfun(@(q) q.Position.r, rep(:), 'UniformOutput', false));
r.psi = cell2mat(arrayfun(@(q) q.Position.psi, rep(:), 'UniformOutput', false));
r.phi = cell2mat(arrayfun(@(q) q.Position.phi, rep(:), 'UniformOutput', false));
r.start = model.start; r.goal = model.end;
end

function [rep, model] = runAuthorsScript()
rep = []; model = [];            %#ok<NASGU> assigned by the script
evalc('NMOPSO');                 % authors' script, unmodified
close all force
end
