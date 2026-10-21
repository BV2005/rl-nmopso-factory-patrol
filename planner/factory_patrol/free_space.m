function f = free_space(L, X, Y)
% True where a patrol point, lane or route may lie (X, Y arrays of world metres).
% With L.cylMargin set (auto layouts): at least L.cylMargin outside every cylinder that NMOPSO is
% given for the obstacles (factory_cylinders), i.e. the same obstacle model the planner sees - the
% cylinders bulge up to ~0.3 m beyond the inflated rectangles, so a rectangle-based test can mark
% points free that NMOPSO can never reach. Otherwise (older layouts, unchanged): at least
% L.freeMargin outside every obstacle rectangle inflated by L.inflate.
persistent key cyl
f = true(size(X));
if isfield(L, 'cylMargin')
    k = [L.name mat2str(L.rects) mat2str([L.inflate L.cell])];
    if ~isequal(k, key), cyl = factory_cylinders(L); key = k; end
    for i = 1:size(cyl, 1)
        f = f & hypot(X - cyl(i, 1), Y - cyl(i, 2)) - cyl(i, 3) >= L.cylMargin;
    end
else
    Ri = L.rects(:, 1:4) + L.inflate * [-1 1 -1 1];
    for i = 1:size(Ri, 1)
        dx = max(max(Ri(i, 1) - X, 0), X - Ri(i, 2)); dy = max(max(Ri(i, 3) - Y, 0), Y - Ri(i, 4));
        f = f & hypot(dx, dy) >= L.freeMargin;
    end
end
end
