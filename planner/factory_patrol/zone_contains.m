function ok = zone_contains(Z, d, p, margin)
% pd_zone_contains() of patrol_drone firmware/core/pd_zone.c for the power diagram in Z
% (factory_zones): true iff world point p is on zone d's side of every other zone's bisector by at
% least `margin` metres. The bisector between d and e is linear,
%   g(p) = 2 p.(seed_e - seed_d) + |seed_d|^2 - |seed_e|^2 - w_d + w_e,   p in d iff g(p) <= 0,
% so |g(p)| / |2 (seed_e - seed_d)| is p's exact perpendicular distance to it.
ok = true;
for e = [1:d-1, d+1:numel(Z.w)]
    nv = 2 * (Z.seeds(e, :) - Z.seeds(d, :));
    b = sum(Z.seeds(d, :).^2) - sum(Z.seeds(e, :).^2) - Z.w(d) + Z.w(e);
    if (nv * p(:) + b) / norm(nv) > -margin, ok = false; return, end
end
end
