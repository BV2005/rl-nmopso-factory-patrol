function PlotModel(~)
% No-op stand-in for the authors' PlotModel.m, used only on parallel workers when the environment
% variable NMOPSO_NOPLOT=1 (see LoadScenario.m): their CreateModel.m plots the terrain on every call,
% which leaves figure-renderer processes (~0.7 GB per worker) behind. Results are unaffected.
end
