"""Read-only display of coarse/fine decisions, including unsuccessful rounds."""
import textwrap
import numpy as np


def draw_search_history(figure, history, status=''):
    figure.clear()
    rounds = sorted({stage['round'] for stage in history})
    axes = figure.subplots(max(1, len(rounds)), 1, squeeze=False).ravel()
    colors = {'coarse': '#268bd2', 'coarse+sides': '#8a55b0', 'fine+global-coarse': '#d97706'}
    for ax, round_id in zip(axes, rounds):
        stages = [stage for stage in history if stage['round'] == round_id]
        last = stages[-1]
        for stage in stages:
            if 'candidate_group_scores' not in stage:
                continue
            x = np.unique(stage['candidate_along_offsets'])
            y = []
            for a in x:
                ids = np.flatnonzero(stage['candidate_along_offsets'] == a)
                i = ids[np.argmin(stage['candidate_objective_scores'][ids])]
                value = stage['candidate_group_scores'][i]
                y.append(value if np.isfinite(value) else np.nan)
            ax.plot(x, y, '.-', lw=1, ms=3, color=colors.get(stage['stage']),
                label=f"{stage['stage']}: {stage['candidate_count']} samples, "
                      f"{stage['descriptor_rows']} new rows, {stage['elapsed_ms']:.0f} ms")
        valley = last.get('valley', {})
        if 'along' in valley:
            ax.axvline(valley['along'], color='red', ls=':',
                       label='valley minimum' if valley.get('clear') else 'lowest sample (unclear)')
        if 'basin_bounds' in valley:
            ax.axvspan(*valley['basin_bounds'], color='green', alpha=.12, label='valley bottom')
        if last.get('fine_bounds') is not None:
            ax.axvspan(*last['fine_bounds'], color='orange', alpha=.12, label='fine window')
        ax.set_title(textwrap.fill(
            f"Round {round_id}: cell {last['cell_width']}x{last['cell_height']}, "
            f"{last['points_per_cell']}/cell, {last['point_count']} anchors | "
            f"valleyDrop={valley.get('relative_depth', float('nan')):.3f}, "
            f"basin ratio={valley.get('basin_ratio', float('nan')):.3f} | {last['reason']}",
            width=125), fontsize=9)
        ax.set(xlabel='Along epipolar offset (warped px)', ylabel='GroupScore')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7, loc='best', ncol=2)
        else:
            ax.text(.5, .5, 'No scoreable candidates (see rejection reason)',
                    ha='center', transform=ax.transAxes)
    figure.suptitle(textwrap.fill('Coarse-to-fine search | '+str(status), width=130), fontsize=10)
    figure.text(.02, .01, 'Compare curves only within a round (anchors change on expansion). '
                'Across envelope selects by Objective; shown ordinate is GroupScore.\n'
                'Only evaluated positions are shown; gaps are not interpolated. Coarse coverage is not exhaustive.', fontsize=8)
    figure.tight_layout(rect=(0, .065, 1, .93))
