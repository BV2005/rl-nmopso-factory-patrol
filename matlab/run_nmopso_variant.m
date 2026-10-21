function run_nmopso_variant(legFile, outDir, seeds, nWorkers, maxIt, warmFile)
% Like run_nmopso_legs.m, but runs matlab/nmopso_variant/NMOPSO.m: the authors' script with MaxIt
% from NMOPSO_MAXIT and, when warmFile is given, warm-start particles (rl_factory_patrol hybrid).
% warmFile (MAT): r, psi, phi [K x M x n] seed paths per leg, cnt [K x 1] (rows used per leg).
% Same seeds as run_nmopso_legs.m (rng(1000 * s + k)), so cold runs differ from the authors' only in MaxIt.
P = setup_paths();
if nargin < 6, warmFile = ''; end
I = load(legFile);
W = []; if ~isempty(warmFile), W = load(warmFile); end
if ~isfolder(outDir), mkdir(outDir); end
if isempty(gcp('nocreate'))
    c = parcluster('Processes'); c.NumWorkers = max(c.NumWorkers, nWorkers); parpool(c, nWorkers);
end
pool = gcp;
paths = [{fullfile(P.here, 'nmopso_variant')}, P.paths];       % the variant shadows the authors' NMOPSO.m
L = factory_layout(P.layout);
F = parallel.FevalFuture.empty; names = {};
for k = 1:size(I.legs, 1)
    w = [];
    if ~isempty(W) && W.cnt(k) > 0
        m = W.cnt(k);
        w.r = squeeze(W.r(k, 1:m, :)); w.psi = squeeze(W.psi(k, 1:m, :)); w.phi = squeeze(W.phi(k, 1:m, :));
        if m == 1, w.r = w.r(:)'; w.psi = w.psi(:)'; w.phi = w.phi(:)'; end
    end
    for s = seeds
        out = fullfile(outDir, sprintf('leg%03d_seed%d.mat', k, s));
        if isfile(out), continue, end
        F(end + 1) = parfeval(pool, @oneLeg, 1, I.legs(k, :), 1000 * s + k, paths, L, maxIt, w); %#ok<AGROW>
        names{end + 1} = out; %#ok<AGROW>
    end
end
fprintf('%d NMOPSO variant jobs (MaxIt %d, warm %d) on %d workers\n', numel(F), maxIt, ~isempty(W), pool.NumWorkers);
for j = 1:numel(F)
    [idx, r] = fetchNext(F);
    save(names{idx}, '-struct', 'r');
    fprintf('%s  %s  %.0f s, %d front points\n', datestr(now, 'HH:MM:SS'), names{idx}, r.time, size(r.front, 2)); %#ok<TNOW1,DATST>
end
end

function r = oneLeg(leg, seed, paths, L, maxIt, w)
for p = numel(paths):-1:1, addpath(paths{p}, '-begin'); end
f = write_leg(L, leg(1:3), leg(4:6));
setenv('NMOPSO_LEG_FILE', f);
setenv('NMOPSO_MAXIT', num2str(maxIt));
wf = '';
if ~isempty(w), wf = [tempname '.mat']; save(wf, '-struct', 'w'); end
setenv('NMOPSO_WARM_FILE', wf);
rng(seed, 'twister');
t0 = tic;
[rep, model] = runScript();
r.time = toc(t0);
delete(f); if ~isempty(wf), delete(wf); end
setenv('NMOPSO_MAXIT', ''); setenv('NMOPSO_WARM_FILE', '');
r.seed = seed; r.leg = leg; r.maxIt = maxIt; r.warm = ~isempty(w);
r.front = [rep.Cost];
r.r = cell2mat(arrayfun(@(q) q.Position.r, rep(:), 'UniformOutput', false));
r.psi = cell2mat(arrayfun(@(q) q.Position.psi, rep(:), 'UniformOutput', false));
r.phi = cell2mat(arrayfun(@(q) q.Position.phi, rep(:), 'UniformOutput', false));
r.start = model.start; r.goal = model.end;
end

function [rep, model] = runScript()
rep = []; model = [];            %#ok<NASGU> assigned by the script
evalc('NMOPSO');                 % matlab/nmopso_variant/NMOPSO.m (first on the path)
close all force
end
