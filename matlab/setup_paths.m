function P = setup_paths()
% Paths of the existing planner and the authors' unmodified NMOPSO repository (nothing is copied or
% edited; this project only calls them). The factory model is factory_patrol's CreateModel.m, which
% must shadow the authors' CreateModel.m, so it goes first on the path.
P.root = 'C:\Users\bhara\Downloads\NMOPSO_repro';
P.fp = fullfile(P.root, 'factory_patrol');
P.authors = fullfile(P.root, 'authors');
P.here = fileparts(mfilename('fullpath'));
P.layout = 'factory2_auto';                 % obstacles read from Downloads\factory2 (factory_from_sdf.m)
P.paths = {fullfile(P.fp, 'noplot'), fullfile(P.root, 'src', 'noplot'), fullfile(P.fp, 'model'), P.fp, P.authors, P.here};
for k = numel(P.paths):-1:1, addpath(P.paths{k}, '-begin'); end
end
