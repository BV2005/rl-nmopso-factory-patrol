function c = factory_cylinders(L)
% Covers every obstacle rectangle, inflated by L.inflate, with cylinders (NMOPSO models obstacles
% as cylinders only). The inflated rectangle is split into k1 x k2 equal cells no larger than
% L.cell; each cell gets the circle through its corners, so the union covers the rectangle.
% With L.thinR set, rectangles narrower than L.cell are instead covered by one row of circles of
% radius L.thinR on their long axis, spaced so that the row still covers the full width (a much
% tighter cover for racks and walls, which keeps the aisles between them open).
% Returns [cx cy R] in world metres.
c = zeros(0, 3);
for i = 1:size(L.rects, 1)
    r = L.rects(i, 1:4) + L.inflate * [-1 1 -1 1];
    w = r(2) - r(1); h = r(4) - r(3);
    if isfield(L, 'thinR') && min(w, h) < L.cell && L.thinR > min(w, h) / 2
        long = max(w, h); half = min(w, h) / 2;
        step = 2 * sqrt(L.thinR^2 - half^2);           % length of the rectangle one circle covers
        k = ceil(long / step);
        s = ((1:k)' - 0.5) * long / k;                  % circle centres along the long axis
        if w >= h, t = [r(1) + s, repmat((r(3) + r(4)) / 2, k, 1)];
        else,      t = [repmat((r(1) + r(2)) / 2, k, 1), r(3) + s]; end
        c = [c; t, repmat(L.thinR, k, 1)]; %#ok<AGROW>
        continue
    end
    k1 = max(1, ceil(w / L.cell)); k2 = max(1, ceil(h / L.cell));
    dx = w / k1; dy = h / k2;
    R = hypot(dx, dy) / 2;
    [I, J] = meshgrid(0:k1-1, 0:k2-1);
    c = [c; r(1) + (I(:) + 0.5) * dx, r(3) + (J(:) + 0.5) * dy, repmat(R, numel(I), 1)]; %#ok<AGROW>
end
end
