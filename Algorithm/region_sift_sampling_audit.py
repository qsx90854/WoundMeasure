"""Read-only inspection of the masked descriptor's discrete sampling weights."""
import numpy as np
from Algorithm import masked_sift_descriptor as masked


def sampling_weights(size, angle):
    radius = int(np.rint(3 * (size / 2.) * np.sqrt(2) * 2.5))
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    theta = np.deg2rad(angle)
    x = (np.cos(theta)*xx + np.sin(theta)*yy) / (1.5*size)
    y = (-np.sin(theta)*xx + np.cos(theta)*yy) / (1.5*size)
    bx, by = x + 1.5, y + 1.5
    gaussian = np.exp(-(x*x+y*y)/8.)
    weights = np.stack([np.maximum(0., 1-np.abs(bx-col)) *
                        np.maximum(0., 1-np.abs(by-row)) * gaussian
                        for row in range(4) for col in range(4)])
    return xx, yy, weights


def inspect_samples(context, point, size, angle, excluded):
    entry = min(context['entries'], key=lambda e: abs(e['size']-size))
    if not np.isclose(entry['size'], size, rtol=1e-5):
        raise ValueError('Descriptor scale is absent from audit context')
    xx, yy, weights = sampling_weights(size, angle)
    origin = np.asarray(context.get('origin', (0, 0)))
    cx, cy = np.rint(np.asarray(point)-origin).astype(int)
    xs, ys = cx+xx, cy+yy
    h, w = context['image_shape']
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xi, yi = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    good = inside & entry['gradient_valid'][yi, xi]
    strength, _ = masked._entry_gradient_polar(entry)
    total = weights.sum(axis=(1, 2))
    observed = (weights*good).sum(axis=(1, 2))
    return dict(x=xs+origin[0], y=ys+origin[1], weights=weights,
                good=good, inside=inside, excluded=inside & excluded[yi, xi],
                strength=strength[yi, xi], coverage=observed/np.maximum(total, 1e-12))


def build_point_audit(debug, index):
    from Algorithm.region_sift_search import frame_context
    cfg = debug['config']
    record = debug['point_metadata'][index]
    results = []
    for side in ('left', 'right'):
        image = debug['left_gray'] if side == 'left' else debug['warped_right_gray']
        mask = debug['left_specular_mask'] if side == 'left' else debug['warped_specular_mask']
        valid = None if side == 'left' else debug['warped_valid_mask']
        roi = debug[side + '_audit_context_roi']
        context = frame_context(image, cfg, None, side, mask, valid, roi=roi)
        origin = context['origin']
        h, w = context['image_shape']
        ox, oy = origin
        excluded = masked.exclusion_mask(mask[oy:oy+h, ox:ox+w], cfg) > 0
        point = record['left_point'] if side == 'left' else record['right_point_warp']
        result = inspect_samples(context, point, record[side+'_scale_px'],
                                 record[side+'_angle_deg'], excluded)
        result['image'] = image
        results.append(result)
    return results


class SamplingAuditWindow:
    """Cell selection stays in this window; it never changes a match."""
    def __init__(self, debug, index):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        self.samples = build_point_audit(debug, index)
        self.debug, self.index = debug, index
        self.cell = None
        self.mass = False
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title('Region-SIFT sampling audit')
        grid = self.fig.add_gridspec(2, 6, height_ratios=(2.4, 1),
            left=.06, right=.96, bottom=.14, top=.85, hspace=.35, wspace=.8)
        self.image_axes = [self.fig.add_subplot(grid[0, :3]), self.fig.add_subplot(grid[0, 3:])]
        self.cell_axes = [self.fig.add_subplot(grid[1, j*2:(j+1)*2]) for j in range(3)]
        self.buttons = []
        for bounds, label, action in (([.18,.035,.23,.045], 'All 16 cells', self.all_cells),
                                      ([.56,.035,.28,.045], 'Weight / gradient mass', self.toggle_mass)):
            button = Button(self.fig.add_axes(bounds), label)
            button.on_clicked(action)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.draw()
        plt.show(block=False)

    def all_cells(self, event=None):
        self.cell = None
        self.draw()

    def toggle_mass(self, event=None):
        self.mass = not self.mass
        self.draw()

    def on_click(self, event):
        if event.inaxes not in self.cell_axes or event.xdata is None or event.ydata is None:
            return
        col, row = int(np.floor(event.xdata+.5)), int(np.floor(event.ydata+.5))
        if 0 <= row < 4 and 0 <= col < 4:
            self.cell = row*4+col
            self.draw()

    def draw(self):
        from matplotlib.patches import Rectangle
        cfg = self.debug['config']
        lc, rc = (sample['coverage'] for sample in self.samples)
        common = np.maximum(0., lc+rc-1.)
        usable = common >= cfg.masked_min_cell_fraction
        label = 'all cells' if self.cell is None else f'cell {self.cell//4+1}:{self.cell%4+1}'
        mode = 'pre-normalization gradient mass (cell gate applied; not final L2)' if self.mass else 'spatial interpolation x Gaussian weight'
        rejected = self.cell is not None and not usable[self.cell]
        status = 'EXCLUDED from pair comparison' if rejected else 'cell gate applied to contributing mass'
        pair_status = 'VALID pair' if self.debug['point_metadata'][self.index]['masked_pair_valid'] else 'INVALID pair: inspection only'
        self.fig.suptitle(f'Point {self.index+1} | {label} | {status} | {pair_status}\n{mode}\n'
            'White boundary: sampled gradient centers | green: valid | magenta: specular mask | red: other invalid\n'
            'Brightness is relative within each side; click a cell below. Yellow = conservative full support.', fontsize=10)
        for side, (ax, sample) in enumerate(zip(self.image_axes, self.samples)):
            ax.clear()
            weights = sample['weights'].sum(axis=0) if self.cell is None else sample['weights'][self.cell]
            footprint = weights > 0
            if self.mass:
                active = np.arange(16)[usable] if self.cell is None else ([self.cell] if usable[self.cell] else [])
                values = sample['weights'][active].sum(axis=0) * sample['strength'] * sample['good']
            else:
                values = weights
            alpha = values / max(float(values.max()), 1e-12)
            rgba = np.zeros((*weights.shape, 4))
            rgba[..., 1] = 1.
            spec = footprint & sample['excluded']
            invalid = footprint & ~sample['good'] & ~spec
            rgba[spec, :3] = (1., 0., 1.)
            rgba[invalid, :3] = (1., .15, .05)
            rgba[..., 3] = np.where(footprint, (.75*alpha if self.mass else .12+.65*alpha), 0.)
            rgba[spec | invalid, 3] = .7
            x, y = sample['x'], sample['y']
            extent = (x.min()-.5, x.max()+.5, y.max()+.5, y.min()-.5)
            ax.imshow(sample['image'], cmap='gray', vmin=0, vmax=255)
            ax.imshow(rgba, extent=extent, interpolation='nearest', origin='upper')
            padded = np.pad(footprint.astype(float), 1)
            ax.contour(np.arange(padded.shape[1])+x.min()-1,
                       np.arange(padded.shape[0])+y.min()-1, padded,
                       levels=[.5], colors='white', linewidths=1.)
            record = self.debug['point_metadata'][self.index]
            prefix = 'left' if side == 0 else 'right'
            point = record['left_point'] if side == 0 else record['right_point_warp']
            radius = record[prefix+'_support_radius_px']
            ax.add_patch(Rectangle(np.asarray(point)-radius, 2*radius, 2*radius,
                                  fill=False, edgecolor='yellow', linestyle=':'))
            ax.set_xlim(point[0]-radius-2, point[0]+radius+2)
            ax.set_ylim(point[1]+radius+2, point[1]-radius-2)
            ax.set_title(('Left' if side == 0 else 'Right (warped)') +
                         f' | valid samples {np.count_nonzero(footprint & sample["good"])}/{np.count_nonzero(footprint)}')
        for ax, values, name in zip(self.cell_axes, (lc, rc, common), ('Left coverage', 'Right coverage', 'Common lower bound')):
            ax.clear()
            ax.imshow(values.reshape(4,4), vmin=0, vmax=1, cmap='viridis')
            for k, value in enumerate(values):
                row, col = divmod(k,4)
                ax.text(col,row,f'{value:.2f}'+(' X' if not usable[k] else ''),
                        ha='center',va='center',color=('black' if value > .55 else 'white'),fontsize=9)
            if self.cell is not None:
                ax.add_patch(Rectangle((self.cell%4-.5,self.cell//4-.5),1,1,
                                      fill=False,edgecolor='yellow',linewidth=2))
            ax.set_xticks(range(4), range(1,5)); ax.set_yticks(range(4), range(1,5))
            ax.set_title(name + f' (X: common < {cfg.masked_min_cell_fraction:.2f})',fontsize=9)
        self.fig.canvas.draw_idle()
