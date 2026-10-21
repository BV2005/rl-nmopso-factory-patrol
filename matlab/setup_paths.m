function P = setup_paths()
% Paths of the factory patrol planner (planner/factory_patrol) and the authors' unmodified NMOPSO
% (third_party/NMOPSO), relative to this repository. The factory model is factory_patrol's
% CreateModel.m, which must shadow the authors' CreateModel.m, so it goes first on the path.
% The world file is worlds/factory2.sdf unless NMOPSO_WORLD is already set.
P.here = fileparts(mfilename('fullpath'));
P.repo = fileparts(P.here);
P.fp = fullfile(P.repo, 'planner', 'factory_patrol');
P.authors = fullfile(P.repo, 'third_party', 'NMOPSO');
P.layout = 'factory2_auto';                 % obstacles read from the world file (factory_from_sdf.m)
if isempty(getenv('NMOPSO_WORLD')), setenv('NMOPSO_WORLD', fullfile(P.repo, 'worlds', 'factory2.sdf')); end
P.paths = {fullfile(P.fp, 'noplot'), fullfile(P.repo, 'planner', 'src', 'noplot'), fullfile(P.fp, 'model'), P.fp, P.authors, P.here};
for k = numel(P.paths):-1:1, addpath(P.paths{k}, '-begin'); end
end
