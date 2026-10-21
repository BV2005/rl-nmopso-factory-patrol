function export_world(outFile)
% Writes the NMOPSO model of the factory2 world (layout factory2_auto, whole hall, patrol band) as
% JSON for the Python environment: hall, scale, bounds in NMOPSO units, obstacle cylinders
% (factory_cylinders.m, the threats CreateModel.m gives NMOPSO), obstacle rectangles, free-space
% settings. Built through factory_patrol's own CreateModel.m on a dummy leg, so it is exactly the
% model NMOPSO sees.
P = setup_paths();
if nargin < 1, outFile = fullfile(P.here, '..', 'data', 'factory2_model.json'); end
L = factory_layout(P.layout);
f = write_leg(L, [0 -8 1.8], [0 8 1.8]);
setenv('NMOPSO_LEG_FILE', f); model = CreateModel(); delete(f);
J.layout = P.layout;
J.world = L.world;
J.hall = L.hall;
J.scale = L.scale;
J.wallMargin = L.wallMargin;
J.zmin = L.zmin; J.zmax = L.zmax; J.zPatrol = L.zPatrol;
J.bounds_units = [model.xmin model.xmax model.ymin model.ymax model.zmin model.zmax];
J.mapsize = [model.MAPSIZE_X model.MAPSIZE_Y];
J.threats_units = model.threats;                 % [x y z R] in NMOPSO units
J.cylinders_m = factory_cylinders(L);            % [x y R] metres
J.rects = L.rects; J.names = L.names;
J.overflown = L.overflown; J.overflownNames = L.overflownNames;
J.cylMargin = L.cylMargin; J.inflate = L.inflate;
d = fileparts(outFile); if ~isfolder(d), mkdir(d); end
fid = fopen(outFile, 'w'); fprintf(fid, '%s', jsonencode(J, 'PrettyPrint', true)); fclose(fid);
fprintf('wrote %s: %d threats\n', outFile, size(model.threats, 1));
end
