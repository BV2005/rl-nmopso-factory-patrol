function [ord, flip, len] = tour_order(pc, s0)
% Closed patrol tour over lane pieces pc = [x1 y1 x2 y2] (each flown end to end): the order and
% direction of the pieces, by nearest neighbour from s0 and then 2-opt on the cycle (straight-line
% links between piece ends). The cycle is rotated to start at the piece start nearest s0.
%   ord(k) = piece flown k-th, flip(k) = flown from its (x2, y2) end; len = pieces + links of the
%   closed cycle (a single piece is flown out and back).
n = size(pc, 1); ord = zeros(0, 1); flip = false(0, 1); len = 0;
if n == 0, return, end
A = pc(:, 1:2); B = pc(:, 3:4); s0 = s0(1:2);
used = false(n, 1); ord = zeros(n, 1); flip = false(n, 1); cur = s0;
for k = 1:n                                            % nearest neighbour
    da = vecnorm(A - cur, 2, 2); db = vecnorm(B - cur, 2, 2); da(used) = inf; db(used) = inf;
    [ma, ia] = min(da); [mb, ib] = min(db);
    if ma <= mb, ord(k) = ia; flip(k) = false; cur = B(ia, :);
    else, ord(k) = ib; flip(k) = true; cur = A(ib, :); end
    used(ord(k)) = true;
end
improved = true;                                       % 2-opt on the cycle
while improved && n > 2
    improved = false;
    [S, E] = ends(A, B, ord, flip);
    for i = 1:n - 1
        for j = i + 1:n
            ip = mod(i - 2, n) + 1; jn = mod(j, n) + 1;
            if ip == j || jn == i, continue, end        % reversing the whole cycle changes nothing
            old = norm(E(ip, :) - S(i, :)) + norm(E(j, :) - S(jn, :));
            new = norm(E(ip, :) - E(j, :)) + norm(S(i, :) - S(jn, :));
            if new < old - 1e-9
                ord(i:j) = flipud(ord(i:j)); flip(i:j) = ~flipud(flip(i:j));
                improved = true; break
            end
        end
        if improved, break, end
    end
end
[S, ~] = ends(A, B, ord, flip);                        % start at the piece start nearest s0
[~, r] = min(vecnorm(S - s0, 2, 2));
ord = circshift(ord, 1 - r); flip = circshift(flip, 1 - r);
[S, E] = ends(A, B, ord, flip);
len = sum(vecnorm(E - S, 2, 2)) + sum(vecnorm(S([2:n 1], :) - E, 2, 2));
end

function [S, E] = ends(A, B, ord, flip)
S = A(ord, :); E = B(ord, :);
S(flip, :) = B(ord(flip), :); E(flip, :) = A(ord(flip), :);
end
