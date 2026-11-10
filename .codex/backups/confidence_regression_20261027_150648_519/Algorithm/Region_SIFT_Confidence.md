# Region-SIFT confidence and mask-aware anchor sampling

First-stage implementation: no new descriptor rows, scale maps or render events
are needed by confidence analysis. Pixel-aligned common-mask verification is
deferred; the current comparison still uses subcell coverage estimates.

## Controls in SIFT parameters

- `confidence_mode`: `legacy` keeps original rejection rules and skips confidence
  analysis; `shadow` keeps those decisions and records the new proposal; `active`
  uses Objective basins, localization width and cell stability instead of the old
  spatial runner-up ratio and GroupScore shoulder-depth gate. The main program
  starts in `shadow`; the library default is `legacy`.
- `masked_sampling_gradient=True`: custom descriptors use single-scale normalized
  smoothing and valid gradient stencils for anchor ranking in the local ROI.
  This is independent of confidence mode. For an exact old-sampling baseline,
  set this to False as well as `confidence_mode=legacy`.
- `confidence_score_band_relative=0.05`: candidates within
  `bestObjective + 0.05*abs(bestObjective)` form the sampled near-best interval.
- `confidence_basin_prominence_relative=0.03`: two local minima are split only if
  the intervening ridge exceeds both minima by this fraction of the larger
  minimum's magnitude. Invalid samples and physical gaps split components.
- `confidence_max_width_px=4`: active rejects wider within-basin intervals.
- `confidence_cell_stability=True`, `confidence_min_cell_stability=0.75`: once at
  the end of search, omit each Region cell and rescore surviving candidates;
  require enough trials to remain in the original Objective basin. These trials
  use texture-plus-missing costs, recompute KEEP/cell weights, and do not repeat
  scale/orientation consistency estimation. Trials are not independent votes.
- `confidence_height_span_max_mm=None`: display only. A positive value adds an
  active-mode height-span veto in the main application. Missing/invalid geometry
  also fails that optional veto. Standalone matcher callers must call
  `attach_height_span` with their geometry and plane to use this application gate.

These initial thresholds are engineering settings, not calibrated probabilities.
Original geometry, coverage, BestG, flat/all-zero evidence and fine-search-boundary
checks remain effective. Active mode never fabricates a result when no candidate
can be scored. Coarse-only terminal results remain uncertain. Fine intervals
touching missing samples or containing larger-than-fine-step gaps are uncertain.

## Logs and data

`[Region-SIFT Confidence]` shows `input` (decision before this confidence pass),
`proposal`, reason codes and elapsed time. In shadow the proposal is diagnostic
only: it cannot change `m_pt` or `reject_reason`. Active can reach different search
rounds, so shadow does not simulate the cost/results of a full active-mode rerun.

Best, legacy second and other-basin candidates show:

- `texture contribution`: raw L2 attributed using the ORIGINAL KEEP indices and
  cell weights. Invalid rows contribute zero here, not a perfect observed match.
- `missing`: the missing-evidence component, including full ceiling distances of
  invalid rows. Texture + missing reconstructs GroupScore (float precision).
- frame penalty, valid pair count, average common coverage, whether the old
  3-pixel-excluded second candidate belongs to the same basin.
- same-basin along span, competing basin ratio, leave-one-cell-out stability.

Small candidate-by-anchor arrays live in `region_debug.evidence_*`; there is no
second copy of all 128-bin histograms. `region_debug.confidence` and
`result.confidence` share the report. `scored_candidate` retains the rigid best
coordinate even when formal `m_pt` is rejected.

Height ranges reuse the main application's `triangulate_point_3d`, including
`correctMatches`, positive-depth checks, the selected height plane and offset.
They describe sampled rigid candidates for one right frame before temporal or
multiframe fusion. They are not confidence intervals. When LocalPlane refinement
changes the match, the rigid interval is marked unavailable for that final result.

The main log separately times candidate-height triangulation. Confidence elapsed
time already includes cell-stability time; do not add that subitem again.
All computations work with diagnostic panels disabled and create no figures.

## Verification and rollback

Targeted tests cover masked-intensity invariance of anchor selection, exact score
attribution including invalid pairs, single/competing basins, missing/irregular
samples, shadow result/descriptor-count equality, active vetoes, settings and
height callback behavior. Real-camera accuracy and block MAE need user data.

A local warm synthetic run (250x300 random image, custom descriptor, fixed size
3.2, 95x5 coarse/fine band, five alternating legacy/shadow trials) used 4004 right
descriptor rows in every trial for both modes. Confidence analysis alone had a
15.8 ms median. Whole-call medians were noisy (legacy 766.46 ms, shadow 632.42 ms)
and do NOT demonstrate a speedup. This is not a camera/production benchmark.

Pre-change copies (including the original test fixture) are in
`.codex/backups/masked_confidence_20261027_130243_816/`, retaining relative paths.
Restore only the affected files after checking for subsequent edits. Newly added
files are `Algorithm/region_sift_confidence.py`, this document and
`tests/test_region_sift_confidence.py`.
