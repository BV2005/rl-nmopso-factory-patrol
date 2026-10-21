function diag_feasibility()
% How often are the random initial paths of the authors' NMOPSO.m feasible on factory legs?
% NMOPSO.m keeps re-sampling its initial swarm until at least one path is feasible, so a rate near
% zero means it never gets past initialisation. Uses the authors' CreateRandomSolution,
% SphericalToCart2 and MyCost with the bounds NMOPSO.m sets, on legs of different length.
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(root, 'authors_1000'));
addpath(here, fullfile(here, 'model'), '-begin');
L = factory_layout();
x = -14.9;                                              % drone 1, lane 2 (through machine_bay_3)
tests = {'full lane through machine bay 3 (26 m)', [x 13 1.8; x -13 1.8]
         'lane split: 13 -> 6.5 (6.5 m, next to the machine)', [x 13 1.8; x 6.5 1.8]
         'lane split: 6.5 -> 0 (6.5 m, open floor)', [x 6.5 1.8; x 0 1.8]
         'lane split: 13 -> 9.75 (3.25 m)', [x 13 1.8; x 9.75 1.8]
         'open lane x = -18 (26 m, no obstacle on the line)', [-18 -13 1.8; -18 13 1.8]};
N = 2000;
rng(1);
for t = 1:size(tests, 1)
    start_w = tests{t, 2}(1, :); goal_w = tests{t, 2}(2, :); %#ok<NASGU>
    f = [tempname '.mat']; save(f, 'start_w', 'goal_w'); setenv('NMOPSO_LEG_FILE', f);
    model = CreateModel(); delete(f);
    nVar = model.n; VarSize = [1 nVar];
    VarMax.r = 3 * norm(model.start - model.end) / nVar; VarMin.r = VarMax.r / 9;
    VarMin.psi = -pi/4; VarMax.psi = pi/4; VarMin.phi = -pi/4; VarMax.phi = pi/4;
    C = zeros(4, N);
    for i = 1:N
        sol = CreateRandomSolution(VarSize, VarMin, VarMax);
        C(:, i) = MyCost(SphericalToCart2(sol, model), model, VarMin);
    end
    feas = all(isfinite(C), 1);
    fprintf('%-52s feasible %5.2f %% | F1=inf (short/clamped segment) %4.1f %% | F2=inf (collision) %4.1f %% | expected tries until 1st feasible swarm of 100: %s\n', ...
        tests{t, 1}, 100 * mean(feas), 100 * mean(~isfinite(C(1, :))), 100 * mean(~isfinite(C(2, :))), ...
        tries(mean(feas)));
end
end

function s = tries(p)
if p == 0, s = 'never (0 of 2000 sampled)'; return, end
q = 1 - (1 - p) ^ 100;          % chance that a swarm of 100 contains a feasible path
s = sprintf('%.1f', 1 / q);
end
