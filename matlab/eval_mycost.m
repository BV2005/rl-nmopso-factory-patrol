function eval_mycost(inFile, outFile)
% Scores solutions with the authors' own SphericalToCart2.m and MyCost.m (unmodified).
% inFile (MAT): legs [K x 6] world metres [ax ay az bx by bz]; leg [S x 1] (1-based leg of each
% solution); r, psi, phi [S x n] (NMOPSO's spherical encoding).
% outFile (MAT): cost [S x 4], and xyz [S x n x 3] (the decoded nodes, NMOPSO units).
P = setup_paths();
I = load(inFile);
L = factory_layout(P.layout);
S = size(I.r, 1); cost = zeros(S, 4); xyz = zeros(S, size(I.r, 2), 3);
for k = 1:size(I.legs, 1)
    f = write_leg(L, I.legs(k, 1:3), I.legs(k, 4:6));
    setenv('NMOPSO_LEG_FILE', f); model = CreateModel(); delete(f);
    VarMin.r = (3 * norm(model.start - model.end) / model.n) / 9;   % as in NMOPSO.m
    for s = find(I.leg(:)' == k)
        sol.r = I.r(s, :); sol.psi = I.psi(s, :); sol.phi = I.phi(s, :);
        pos = SphericalToCart2(sol, model);
        cost(s, :) = MyCost(pos, model, VarMin)';
        xyz(s, :, :) = cat(3, pos.x, pos.y, pos.z);
    end
end
save(outFile, 'cost', 'xyz');
end
