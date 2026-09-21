"""Cache unchanged UI while navigating a pair of connected image axes."""
from contextlib import contextmanager


@contextmanager
def _animation_state(artists, animated):
    states = [(artist, artist.get_animated()) for artist in artists]
    try:
        for artist, _ in states:
            artist.set_animated(animated)
        yield
    finally:
        for artist, previous in states:
            artist.set_animated(previous)


class PairedViewRenderer:
    """GUI-thread renderer. Rebuild on non-view changes or pair switches.

    Whole-canvas backgrounds preserve connections and unclipped annotations
    outside axes bounds. Only the active pair is redrawn during navigation.
    """

    def __init__(self, figure, pairs, overlays=()):
        self.figure = figure
        self.pairs = tuple(tuple(pair) for pair in pairs)
        self.overlays = tuple(overlays)
        self.background = None
        self.cached_pair = None
        self.geometry = None

    def invalidate(self):
        self.background = None
        self.cached_pair = None

    def _draw_pair(self, pair):
        for ax in pair:
            if ax.get_visible():
                # Preserve zorder and draw each child exactly once, including
                # normally animated images and height-profile annotations.
                with _animation_state(ax.get_children(), False):
                    ax.draw(self.figure.canvas.get_renderer())

    def draw(self, pair_index=None):
        canvas = self.figure.canvas
        geometry = (tuple(self.figure.bbox.bounds), self.figure.dpi)
        rebuild = (self.background is None or self.cached_pair != pair_index
                   or self.geometry != geometry or pair_index is None)
        axes = [ax for pair in self.pairs for ax in pair]
        if rebuild:
            with _animation_state(axes + list(self.overlays), True):
                canvas.draw()
            if pair_index is not None:
                for index, pair in enumerate(self.pairs):
                    if index != pair_index:
                        self._draw_pair(pair)
                self.background = canvas.copy_from_bbox(self.figure.bbox)
                self.cached_pair = pair_index
                self.geometry = geometry
            else:
                self.invalidate()
        else:
            canvas.restore_region(self.background)
        if pair_index is None:
            for pair in self.pairs:
                self._draw_pair(pair)
        else:
            self._draw_pair(self.pairs[pair_index])
        for artist in self.overlays:
            if artist.get_visible():
                self.figure.draw_artist(artist)
        canvas.blit(self.figure.bbox)
        return rebuild
