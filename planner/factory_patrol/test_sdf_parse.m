% Check: factory_from_sdf on Downloads/factory2 against the hand-typed 'storage_assembly' layout.
W = factory_from_sdf(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'worlds', 'factory2.sdf'), 1.0);
L = factory_layout('storage_assembly');
fprintf('hall  sdf %s   hand %s   roof %.2f\n', mat2str(W.hall, 4), mat2str(L.hall, 4), W.roof);
fprintf('skipped: %s\n', strjoin(cellfun(@(a, b) [a ' (' b ')'], W.skipped(:, 1), W.skipped(:, 2), 'UniformOutput', false), ', '));
[~, ia, ib] = intersect(W.names, L.names);
fprintf('obstacles: sdf %d, hand %d, common %d, max abs difference %.3g m\n', numel(W.names), numel(L.names), ...
        numel(ia), max(abs(W.rects(ia, :) - L.rects(ib, :)), [], 'all'));
[~, ia, ib] = intersect(W.overflownNames, L.overflownNames);
fprintf('flown over: sdf %s, hand %s, max abs difference %.3g m\n', strjoin(W.overflownNames, ' '), ...
        strjoin(L.overflownNames, ' '), max(abs(W.overflown(ia, :) - L.overflown(ib, :)), [], 'all'));
