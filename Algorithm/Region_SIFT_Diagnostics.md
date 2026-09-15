# Region-SIFT ambiguity diagnostics

Enable Region-SIFT and click a measurement point (or use coordinate replay).
The diagnostics window opens automatically after an individual measurement,
including score/ratio rejection when candidate scores exist. It reuses one
window. Block batches and continuous auto measurement do not auto-open it.
The **SIFT Debug** button next to **SIFT 參數** reopens the most recently displayed
Region-SIFT result. Reveal the blue control panel if these buttons are hidden.

- Top left: each thin curve is one across-epipolar offset; the black curve is
  the minimum across offsets. Star = best; triangle = second. Shading is the
  projection of the configured exclusion radius, not a detected basin boundary.
- Top right: lower scores are blue, higher scores yellow, invalid candidates
  gray. The dashed circle is the actual two-dimensional exclusion radius.
  Objective and GroupScore can be switched; best/second markers always identify
  the candidates ranked by the matcher's Objective.
- Middle: left reference and warped-right best/second at identical pixel scale.
  A red cross marks P; click within 15 screen pixels of an anchor to inspect it.
  Toggle IDs, shared-scale L2 colors, KEEP/TRIM, or the selected support bound.
  Filled means KEEP; hollow means TRIM only when KEEP/TRIM is enabled.
  TRIM still contributes to the balanced all-cell score.
- Bottom bars compare every anchor's L2 distance for best versus second.
  Last anchor is P. Details show selected size, angle, support and reliability.
- Candidate heights use existing triangulation/correction and the currently
  selected height plane (including custom-plane/offset rules). They are
  single-frame diagnostic estimates, without temporal fusion; rejection is
  never converted into a successful measurement. Missing/invalid geometry is NaN.

A single broad valley suggests positional uncertainty; separated valleys
suggest competing appearances. A boundary minimum suggests checking search
coverage. A sharp minimum alone does not prove correctness. Compare candidate
height differences with the application's acceptable height error.

No matcher thresholds or selection rules are changed by this viewer. It uses
cached candidate scores and copies only second-candidate point/frame data,
without recomputing SIFT descriptors.

## GT height comparison

Enter the selected block's height in **GT mm**, then press Enter or **Evaluate GT**.
This explicitly evaluates one extra candidate; it does not enter the matcher
ranking or turn a failed measurement into a successful one. On each new
measurement the previous GT overlay is cleared; confirm the height again.

The green diamond is the GT height projected using this measurement's RT and
the selected signed height plane/offset. It is not independent pixel truth.
Its exact warped position is scored with the same anchors, scale/orientation
selection, support/specular gates, descriptor and scoring rules as the matcher.
All anchors still share one displacement, even if their supports cross steps.

- INSIDE/OUTSIDE refers to the configured search band. Outside positions can
  still be scored for diagnosis when their image/support remains valid.
- Nearest sample distance and Objective refer to the original discrete grid;
  they are separate from the newly evaluated exact GT score. An infinite
  nearest-sample score means that original sample was not scoreable.
- GT-BestObj < 0 means the GT prediction scores better than the chosen best.
- Invalid GT positions display the failing stage and anchor IDs. No mask is
  bypassed to manufacture a score. Scale-support diagnostics additionally
  report anchors that become valid when the specular mask is removed.
- The fourth crop shows the GT region at the same pixel scale. The green line
  on the per-anchor bars is its exact L2 distance; the shared color scale now
  includes best, second and GT. The score curve and 2D map include the GT
  marker even outside the original search band.

GT evaluation builds its own right-image frame maps on demand, so pressing
Evaluate GT may briefly take additional computation time. Ordinary matching
does not incur this extra descriptor computation.
