function f = write_leg(L, a, b)
% Leg file read by factory_patrol/model/CreateModel.m: start and goal in world metres, layout name.
% No zone limits or zone walls: the benchmark leg may use the whole hall (minus the wall margin) in
% the patrol band.
start_w = a; goal_w = b; layout = L.name; %#ok<NASGU>
f = [tempname '.mat'];
save(f, 'start_w', 'goal_w', 'layout');
end
