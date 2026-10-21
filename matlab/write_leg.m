function f = write_leg(L, a, b, extra)
% Leg file read by factory_patrol/model/CreateModel.m: start and goal in world metres, layout name.
% No zone limits or zone walls: the benchmark leg may use the whole hall (minus the wall margin) in
% the patrol band. extra (optional, [m x 3] metres [x y R]): additional obstacle cylinders, passed as
% CreateModel.m's extra_threats (e.g. an unexpected obstacle).
start_w = a; goal_w = b; layout = L.name; %#ok<NASGU>
f = [tempname '.mat'];
if nargin > 3 && ~isempty(extra)
    extra_threats = extra; %#ok<NASGU>
    save(f, 'start_w', 'goal_w', 'layout', 'extra_threats');
else
    save(f, 'start_w', 'goal_w', 'layout');
end
end
