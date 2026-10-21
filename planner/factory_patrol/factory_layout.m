function L = factory_layout(name)
% Factory geometry in world ENU metres plus the patrol set-up used by the NMOPSO demo.
%   factory_layout()            or 'patrol_drone': patrol_drone sim/worlds/factory.sdf (40 x 30 x 6 m),
%                               one home per drone inside its own zone
%   factory_layout('factory2')  Downloads/factory.sdf (30 x 20 x 12 m, 4 machines), all drones start
%                               from a shared launch area and fly a transit leg to their zone
%   factory_layout('storage_assembly')  Downloads/factory2 (same hall; storage racks, partition,
%                               assembly machines), shared launch area, explicit aisle lanes
if nargin < 1 || isempty(name), name = 'patrol_drone'; end
L.name = name;
L.wallMargin = 1.5;               % keep waypoints this far from the walls
L.inflate = 0.4;                  % obstacle inflation (drone half-size ~0.3 m + 0.1 m)
L.cell = 2.0;                     % max cylinder cell size when covering a rectangle
% patrol altitude band (m above the floor) and nominal altitude
L.zmin = 1.6; L.zmax = 2.0; L.zPatrol = 1.8;
% NMOPSO works in decimetres (1 unit = 0.1 m): its MyCost.m hard-codes drone size 1 and danger
% distance 10 units, i.e. 0.1 m and 1.0 m here
L.scale = 10;
L.laneSpacing = 4.0;              % 2 m sensor reach on each side
L.maxLeg = 6.5;                   % lanes split into legs of at most this length
L.sharedStart = false;            % true: homes form one launch area outside the zones
L.departDelay = [0 0 0];          % s each drone hovers after take-off before leaving (shared start)
L.freeMargin = 1.0;               % a patrol / detour point is free this far outside every inflated obstacle
L.overflown = zeros(0, 5);        % obstacles low enough to fly over (not given to NMOPSO)
L.overflownNames = {};

switch name
    case 'patrol_drone'
        % Obstacles are the static boxes of factory.sdf. The two movable 1.5 m boxes are included too:
        % the patrol flies at 1.6-2.0 m, below the tops of the racking (2.2 m) and machines (2.0-2.5 m).
        % [xmin xmax ymin ymax top_z], from factory.sdf
        L.rects = [ -10.5  -9.5   -8.0  -2.0  2.2    % racking_row_a
                    -10.5  -9.5    2.0   8.0  2.2    % racking_row_b
                      4.5   5.5   -8.0  -2.0  2.2    % racking_row_c
                      4.5   5.5    2.0   8.0  2.2    % racking_row_d
                     12.0  16.0  -12.0  -8.0  2.5    % machine_bay_1
                     12.0  16.0    8.0  12.0  2.5    % machine_bay_2
                    -16.5 -13.5    8.5  11.5  2.0    % machine_bay_3
                     -6.6  -5.4  -10.6  -9.4  1.5    % dyn_obstacle_1 (static in this demo)
                      8.4   9.6    3.4   4.6  1.5];  % dyn_obstacle_2 (static in this demo)
        L.names = {'racking_row_a', 'racking_row_b', 'racking_row_c', 'racking_row_d', 'machine_bay_1', ...
                   'machine_bay_2', 'machine_bay_3', 'dyn_obstacle_1', 'dyn_obstacle_2'};
        L.hall = [-20 20 -15 15];         % inner wall faces
        L.roof = 6.0;
        xs = linspace(L.hall(1), L.hall(2), 4);
        L.zones = [xs(1:3)', xs(2:4)', repmat(L.hall(3:4), 3, 1)];   % [x0 x1 y0 y1] per drone
        L.homes = [-13.3 -13.0; 0.0 -13.0; 9.0 -13.0];    % take-off / landing points (inside own zone)
    case 'factory2'
        % Downloads/factory.sdf: 30 x 20 m hall (walls 0.3 m thick at x = +-15, y = +-10, ceiling at 12 m),
        % four machines 3-4 m tall, so the 1.6-2.0 m patrol has to fly around all of them
        L.rects = [ -2.0   2.0   0.5   3.5  3.0     % machine_A (blue)
                    -8.5  -5.5   3.5   6.5  4.0     % machine_B (red)
                     5.5   8.5  -7.0  -3.0  3.0     % machine_C (green)
                     4.5   9.5   3.0   5.0  3.0];   % machine_D (yellow)
        L.names = {'machine_A', 'machine_B', 'machine_C', 'machine_D'};
        L.hall = [-14.85 14.85 -9.85 9.85];          % inner wall faces
        L.roof = 11.95;
        xs = linspace(L.hall(1), L.hall(2), 4);
        L.zones = [xs(1:3)', xs(2:4)', repmat(L.hall(3:4), 3, 1)];
        % shared launch area at the south wall, pads 1.5 m apart; the middle zone's lanes start
        % north of it so no patrol lane passes over the pads
        L.sharedStart = true;
        L.homes = [-1.5 -8.0; 0.0 -8.0; 1.5 -8.0];
        L.launchArea = [-2.5 2.5 -8.6 -7.4];         % [xmin xmax ymin ymax] around the pads
        L.zones(2, 3) = L.launchArea(4) + 1.0;       % middle zone starts 1 m north of the pads
        L.corridorY = -7.0;                          % each drone leaves / returns straight north of its
                                                     % own pad up to this y, so no transit crosses a neighbour's pad
        L.departDelay = [0 6 0];                     % the middle drone lets its neighbours leave first
    case {'storage_assembly', 'factory2_voronoi'}
        % Downloads/factory2: 30 x 20 x 12 m hall (walls at x = +-15, y = +-10, invisible ceiling at 12 m);
        % storage area (west) with four 3.5 m racks, a 3.5 m partition at x = 0, assembly area (east).
        % Obstacles at most 1.0 m tall are flown over (patrol 1.6-2.0 m): loading dock, conveyor, floor lines.
        L.rects = [  -0.15   0.15  -4.0   6.0   3.5     % central_partition
                    -12.2  -11.8   -6.0   2.0   3.5     % rack_row_1
                     -9.7   -9.3   -6.0   2.0   3.5     % rack_row_2
                     -7.2   -6.8   -6.0   2.0   3.5     % rack_row_3
                     -4.7   -4.3   -6.0   2.0   3.5     % rack_row_4
                      3.5    6.5    3.75  6.25  2.5     % cnc_machine
                      8.75  11.25   4.0   6.0   2.0     % lathe_machine
                      5.0    9.0   -0.75  0.75  1.2     % assembly_workbench
                      2.5    5.5   -8.0  -6.0   1.5     % packing_station
                      9.0   11.0   -8.0  -6.0   2.0];   % shipping_crate_stack
        L.names = {'central_partition', 'rack_row_1', 'rack_row_2', 'rack_row_3', 'rack_row_4', 'cnc_machine', ...
                   'lathe_machine', 'assembly_workbench', 'packing_station', 'shipping_crate_stack'};
        L.overflown = [ -14.0 -10.0   5.5   8.5   1.0     % loading_dock
                          4.0  12.0  -4.5  -3.5   0.8];   % conveyor_belt
        L.overflownNames = {'loading_dock', 'conveyor_belt'};
        L.hall = [-14.85 14.85 -9.85 9.85];
        L.roof = 11.95;
        % the aisles between the racks are 2.1 m wide: points 0.3 m outside the inflated racks count as
        % free, and thin obstacles are covered by a row of R = 0.7 m circles instead of 2 m cells
        L.freeMargin = 0.3;
        L.thinR = 0.7;
        % zones: storage aisles | both sides of the partition | assembly area
        L.zones = [ L.hall(1) -4.5  L.hall(3:4)
                    -4.5       3.0  L.hall(3:4)
                     3.0  L.hall(2) L.hall(3:4)];
        % explicit lanes: axis = direction the lane runs along, at = lane positions across it
        L.lanes = { struct('axis', 'y', 'at', [-13.3 -10.75 -8.25 -5.75]), ...   % west wall | aisles 1-2, 2-3, 3-4
                    struct('axis', 'y', 'at', [-2.2 1.6]), ...                   % either side of the partition
                    struct('axis', 'x', 'at', [-3.4 2.25 7.85]) };               % between the machine rows
        L.sharedStart = true;
        L.homes = [-2.5 -8.0; -1.0 -8.0; 0.5 -8.0];  % pads 1.5 m apart, south of the partition
        L.launchArea = [-3.2 1.2 -8.6 -7.4];
        L.zones(2, 3) = L.launchArea(4) + 1.0;       % middle zone starts 1 m north of the pads
        L.corridorY = -7.0;
        L.departDelay = [0 6 0];
        if strcmp(name, 'factory2_voronoi')
            % same world and launch area; zones from a balanced power diagram (factory_zones.m)
            % over seeds spread across the whole factory, instead of hand-drawn strips
            L = rmfield(L, 'lanes');
            L.voronoi = true;
            % patrol lane network of the whole world [x1 y1 x2 y2]: the rack aisles and both sides of
            % the partition (running north-south), the lanes between the machine rows (east-west).
            % Lanes passing the launch area start 1.5 m north of it, so no lane runs over the pads.
            xs = [-13.3 -10.75 -8.25 -5.75 -2.2 1.6];
            nearPads = abs(xs - min(max(xs, L.launchArea(1)), L.launchArea(2))) < 1.5;
            yLo = L.hall(3) + 2 + nearPads * (L.launchArea(4) + 1.5 - (L.hall(3) + 2));
            L.laneNet = [ [xs; yLo; xs; repmat(L.hall(4) - 2, 1, numel(xs))]'
                          [repmat(3.5, 3, 1), [-3.4; 2.25; 7.85], repmat(L.hall(2) - 2, 3, 1), [-3.4; 2.25; 7.85]] ];
            L.zones = repmat(L.hall, 3, 1);           % replaced by the zone bounding boxes
            L.zoneMargin = 1.0;                       % lane pieces stay this far inside the bisectors
            L.wallR = 0.5; L.wallSpacing = 0.6;       % zone-boundary cylinders given to NMOPSO
            L.gridRes = 0.2;                          % routing grid between lane pieces
            L.routeMargin = 0.6;                      % routes stay this far inside the bisectors
            % transit layer: all pads lie in one zone, so the other drones' transits must cross it;
            % transits fly at 2.9-3.3 m (patrol band 1.6-2.0 m, racks and partition 3.5 m tall), with
            % vertical climbs / descents over the own pad and the own zone's entry point. A drone whose
            % pad lies in its own zone stays in the patrol band (plan_factory_patrol).
            L.zTransit = 3.1; L.zTransitBand = [2.9 3.3];
        end
    case 'factory2_auto'
        % 'factory2_voronoi' without hand-set geometry: obstacles, fly-overs, hall and roof are read
        % from the world file (factory_from_sdf.m), and the patrol lanes are generated from the free
        % space (auto_lanes.m, called by plan_factory_patrol). Launch area, heights and margins are
        % the settings of 'factory2_voronoi'.
        L = factory_layout('factory2_voronoi');
        L.name = name;
        L.world = getenv('NMOPSO_WORLD');
        if isempty(L.world), L.world = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'worlds', 'factory2.sdf'); end
        W = factory_from_sdf(L.world, L.zmin - 0.6);  % tops <= 1.0 m are flown over
        L.hall = W.hall; L.roof = W.roof;
        L.rects = W.rects; L.names = W.names;
        L.overflown = W.overflown; L.overflownNames = W.overflownNames;
        L = rmfield(L, 'laneNet');
        L.autoLanes = true;
        L.sensorReach = 2.0;                         % coverage: floor within 2 m, line of sight
        L.laneStep = 0.5;                            % candidate lane spacing
        L.laneMinLen = 2.0;
        L.laneTurnCost = 4.0;                        % metres charged per extra lane (fewer, longer lanes)
        L.laneMinGain = 1.0;                         % stop below 1 m^2 of new floor per metre
        L.cylMargin = 0.25;                          % lanes / routes / points: 0.25 m outside NMOPSO's
                                                     % obstacle cylinders (drone size 0.1 m + slack)
        L.zones = repmat(L.hall, 3, 1);
    case 'factory2_auto_full'
        % 'factory2_auto' with the remaining hand-set decisions replaced by algorithms:
        %   starts        - the scenario's (the benchmark's D0-D2 starts), no hand-placed launch area
        %   zone -> drone - optimal assignment (all permutations) minimising start-to-seed distance
        %   transits      - NMOPSO legs in the patrol band from the start to the zone and back; no
        %                   transit layer, corridors or hand-set take-off delays: departure delays come
        %                   from a timed conflict check (nmopso_demo/schedule_departures.py)
        %   path choice   - the shortest of the nearly safest Pareto paths
        % Left as settings (physical parameters, not placements): sensor reach, safety margins, patrol
        % band, speed.
        L = factory_layout('factory2_auto');
        L.name = name;
        L.homes = [-8 7; -4 7; 0 7];
        L.sharedStart = false;
        L = rmfield(L, {'launchArea', 'corridorY', 'zTransit', 'zTransitBand'});
        L.departDelay = [0 0 0];                     % filled in by the conflict check
        L.directTransit = true;
        L.assignZones = 'optimal';
        L.select = 'short-safe';
        L.tourOrder = '2opt';                        % lane pieces ordered as a 2-opt tour
        L.balanceOn = 'tour';                        % zones balanced on estimated flight, not lane length
    otherwise
        error('factory_layout: unknown layout ''%s''', name);
end
end
