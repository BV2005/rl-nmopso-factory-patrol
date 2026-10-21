%{
Factory patrol model in the format of the authors' CreateModel.m, for their unmodified NMOPSO.m.
NMOPSO.m calls CreateModel() without arguments, so the leg to plan (start and goal, world ENU
metres) is read from the MAT file named by the environment variable NMOPSO_LEG_FILE.
Optional fields: layout (factory_layout name), xlim_w / ylim_w (limits of the drone's zone),
extra_threats ([x y R] metres: cylinders lining a Voronoi zone's boundary, added to the obstacles).
Units: 1 unit = 0.1 m (see factory_layout.m); the floor is flat (H = 0), z is height above floor.
%}

function model=CreateModel()

    leg = load(getenv('NMOPSO_LEG_FILE'));        % fields: start_w, goal_w ([x y z] metres)
    if isfield(leg, 'layout'), L = factory_layout(leg.layout); else, L = factory_layout(); end
    s = L.scale;
    toU = @(p) [(p(1) - L.hall(1)) * s + 1; (p(2) - L.hall(3)) * s + 1; p(3) * s];

    MAPSIZE_X = round((L.hall(2) - L.hall(1)) * s) + 1;
    MAPSIZE_Y = round((L.hall(4) - L.hall(3)) * s) + 1;
    H = zeros(MAPSIZE_Y, MAPSIZE_X);              % flat factory floor
    [X,Y] = meshgrid(1:MAPSIZE_X,1:MAPSIZE_Y);

    % Threats as cylinders [x y z R]: each obstacle rectangle, inflated, covered by cylinders
    c = factory_cylinders(L);                     % [cx cy R] metres
    if isfield(leg, 'extra_threats')              % zone-boundary cylinders (plan_factory_patrol)
        c = [c; leg.extra_threats];
    end
    threats = [ (c(:,1) - L.hall(1)) * s + 1, (c(:,2) - L.hall(3)) * s + 1, zeros(size(c,1),1), c(:,3) * s ];

    % Map limits (waypoints are clamped to these by SphericalToCart2): wall margin applied
    xmin = 1 + L.wallMargin * s;
    xmax = MAPSIZE_X - L.wallMargin * s;
    ymin = 1 + L.wallMargin * s;
    ymax = MAPSIZE_Y - L.wallMargin * s;
    if isfield(leg, 'xlim_w')                     % restrict the leg to its drone's zone
        xmin = max(xmin, (leg.xlim_w(1) - L.hall(1)) * s + 1);
        xmax = min(xmax, (leg.xlim_w(2) - L.hall(1)) * s + 1);
    end
    if isfield(leg, 'ylim_w')
        ymin = max(ymin, (leg.ylim_w(1) - L.hall(3)) * s + 1);
        ymax = min(ymax, (leg.ylim_w(2) - L.hall(3)) * s + 1);
    end
    zmin = L.zmin * s;
    zmax = L.zmax * s;
    if isfield(leg, 'zlim_w')                     % altitude band of this leg (e.g. transit layer)
        zmin = leg.zlim_w(1) * s;
        zmax = leg.zlim_w(2) * s;
    end

    % Start and end position
    start_location = toU(leg.start_w(:));
    end_location = toU(leg.goal_w(:));

    % Number of path nodes (not including the start position (start node))
    n=10;

    model.start=start_location;
    model.end=end_location;
    model.n=n;
    model.xmin=xmin;
    model.xmax=xmax;
    model.zmin=zmin;
    model.ymin=ymin;
    model.ymax=ymax;
    model.zmax=zmax;
    model.MAPSIZE_X = MAPSIZE_X;
    model.MAPSIZE_Y = MAPSIZE_Y;
    model.X = X;
    model.Y = Y;
    model.H = H;
    model.threats = threats;
end
