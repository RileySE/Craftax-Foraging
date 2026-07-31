"""Plot the excitatory/inhibitory balance of a signed connectome constraint file.

Each (pre-synaptic block, post-synaptic block) pair of the constraint matrix
holds rnn_units_per_type^2 weights, some non-zero. This renders the percentage
of those non-zero weights that are positive, as a grid with the pre-synaptic
block on Y and the post-synaptic block on X.

Assumes the file was generated with transmitter signs (the default of
generate_connectome_constraints.py). An unsigned file plots as a uniformly
100%-positive grid, which is obvious at a glance.

    python utils/plot_constraint_sign_balance.py constraints.npy
    python utils/plot_constraint_sign_balance.py constraints.npy --dark --min_nonzero 4
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# Diverging blue (inhibitory) <-> red (excitatory) with a neutral band across
# the balanced middle, binned rather than continuous: a continuous ramp puts
# near-identical pale colours on either side of 50% (OKLab dE 10.4, under the
# 15 floor for telling two colours apart), whereas every bin colour below is
# >= 15 from every bin colour of the opposite arm and >= 8 under simulated
# protanopia/deuteranopia. Each arm's steps are >= 0.06 apart in OKLCH
# lightness so the ordering reads as a ramp.
#
# The light neutral is the palette's axis grey rather than its lighter
# diverging midpoint: this plot has a third state (block pairs with no non-zero
# weights at all) drawn as empty, and the lighter midpoint sits only dE 3.0
# from that emptiness, which would make "balanced" and "absent" the same cell.
BOUNDARIES = [0., 20., 30., 40., 60., 70., 80., 100.]
THEME = {
    'light': {
        'colors': ['#1c5cab', '#2a78d6', '#5598e7', '#c3c2b7',
                   '#dd716a', '#c74845', '#9e3432'],
        'empty': '#f9f9f7',
        'surface': '#fcfcfb',
        'ink': '#0b0b0b',
        'ink_secondary': '#52514e',
        'ink_muted': '#898781',
        'axis': '#c3c2b7',
    },
    # Selected for the dark surface, not flipped: the steps move so that the
    # loud end of each arm is the light end, and every pair was re-checked
    # against the dark surface and the dark neutral.
    'dark': {
        'colors': ['#6da7ec', '#3987e5', '#256abf', '#383835',
                   '#b13f3c', '#d75853', '#e4857e'],
        'empty': '#0d0d0d',
        'surface': '#1a1a19',
        'ink': '#ffffff',
        'ink_secondary': '#c3c2b7',
        'ink_muted': '#898781',
        'axis': '#383835',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot per-block-pair sign balance of a signed connectome constraint file.')
    parser.add_argument('constraint_file', help='Constraint .npy from generate_connectome_constraints.py.')
    parser.add_argument('-o', '--outfile', default=None,
                        help='Output image path (default: <constraint_file>_sign_balance.png).')
    parser.add_argument('--min_nonzero', type=int, default=1,
                        help='Draw a block pair only if it has at least this many non-zero '
                             'weights. A pair with one weight can only read 0%% or 100%%, so '
                             'raising this drops the noisiest cells. (default: %(default)s)')
    parser.add_argument('--dark', action='store_true',
                        help='Render on the dark surface instead of the light one.')
    parser.add_argument('--csv', nargs='?', const=True, default=None, metavar='PATH',
                        help='Also write the per-block-pair numbers the plot encodes as a '
                             'CSV, so the values are readable without reading colour. '
                             'Bare --csv writes alongside the image.')
    parser.add_argument('--title', default=None,
                        help='Plot title (default: the constraint file name).')
    parser.add_argument('--dpi', type=int, default=200, help='Output DPI (default: %(default)s).')
    return parser.parse_args()


def block_pair_percent_positive(constraints, min_nonzero):
    """Percentage of non-zero weights that are positive, per (pre, post) block
    pair. Pairs with fewer than min_nonzero non-zero weights come back as NaN."""
    n_nonzero = np.count_nonzero(constraints, axis=(2, 3))
    n_positive = (constraints > 0).sum(axis=(2, 3))
    with np.errstate(invalid='ignore'):
        percent = 100. * n_positive / n_nonzero
    percent = np.where(n_nonzero >= max(min_nonzero, 1), percent, np.nan)
    return percent, n_nonzero, n_positive


def main():
    args = parse_args()
    theme = THEME['dark' if args.dark else 'light']

    constraints = np.load(args.constraint_file)
    if constraints.ndim != 4:
        raise SystemExit(
            f'expected a 4D (pre_type, post_type, pre_unit, post_unit) constraint matrix, '
            f'got shape {constraints.shape}')
    n_pre, n_post = constraints.shape[:2]

    percent, n_nonzero, n_positive = block_pair_percent_positive(constraints, args.min_nonzero)
    drawn = int(np.isfinite(percent).sum())
    total_nonzero = int(n_nonzero.sum())
    total_positive = int(n_positive.sum())
    overall = 100. * total_positive / total_nonzero if total_nonzero else float('nan')

    print(f'{n_pre} x {n_post} block pairs, {drawn} drawn '
          f'({100. * drawn / (n_pre * n_post):.1f}%), {n_pre * n_post - drawn} empty')
    print(f'{total_nonzero} non-zero weights, {overall:.2f}% positive / {100 - overall:.2f}% negative')

    cmap = ListedColormap(theme['colors'])
    cmap.set_bad(theme['empty'])
    norm = BoundaryNorm(BOUNDARIES, cmap.N)

    fig, ax = plt.subplots(figsize=(8.4, 7.2), dpi=args.dpi)
    fig.patch.set_facecolor(theme['surface'])
    ax.set_facecolor(theme['empty'])

    image = ax.imshow(np.ma.masked_invalid(percent), cmap=cmap, norm=norm,
                      origin='upper', interpolation='nearest', aspect='equal',
                      rasterized=True)

    title = args.title or os.path.basename(args.constraint_file)
    ax.set_title(title, color=theme['ink'], fontsize=13, pad=30, loc='left')
    subtitle = (f'{overall:.1f}% of {total_nonzero:,} non-zero weights are positive · '
                f'{drawn:,} of {n_pre * n_post:,} block pairs populated')
    if args.min_nonzero > 1:
        subtitle += f' (>= {args.min_nonzero} weights)'
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=theme['ink_secondary'],
            fontsize=9, va='bottom')

    ax.set_xlabel('post-synaptic block', color=theme['ink_secondary'], fontsize=10, labelpad=8)
    ax.set_ylabel('pre-synaptic block', color=theme['ink_secondary'], fontsize=10, labelpad=8)
    ax.tick_params(colors=theme['ink_muted'], labelsize=8, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(theme['axis'])
        spine.set_linewidth(0.8)

    bar = fig.colorbar(image, ax=ax, boundaries=BOUNDARIES, ticks=BOUNDARIES,
                       spacing='proportional', shrink=0.82, pad=0.03)
    bar.set_label('% of non-zero weights positive (excitatory)',
                  color=theme['ink_secondary'], fontsize=9, labelpad=10)
    bar.ax.tick_params(colors=theme['ink_muted'], labelsize=8, length=3, width=0.8)
    bar.outline.set_edgecolor(theme['axis'])
    bar.outline.set_linewidth(0.8)
    # Name the two poles, so the direction is never colour-alone.
    bar.ax.text(1.06, 0.02, 'inhibitory', transform=bar.ax.transAxes, rotation=90,
                color=theme['ink_muted'], fontsize=8, va='bottom', ha='left')
    bar.ax.text(1.06, 0.98, 'excitatory', transform=bar.ax.transAxes, rotation=90,
                color=theme['ink_muted'], fontsize=8, va='top', ha='left')

    empty_label = 'no non-zero weights'
    if args.min_nonzero > 1:
        empty_label = f'fewer than {args.min_nonzero} non-zero weights'
    legend = ax.legend(
        handles=[Patch(facecolor=theme['empty'], edgecolor=theme['axis'],
                       linewidth=0.8, label=empty_label)],
        loc='upper left', bbox_to_anchor=(0, -0.09), frameon=False, fontsize=9,
        handlelength=1.4, handleheight=1.4, borderpad=0)
    for text in legend.get_texts():
        text.set_color(theme['ink_secondary'])

    outfile = args.outfile or os.path.splitext(args.constraint_file)[0] + '_sign_balance.png'
    fig.savefig(outfile, dpi=args.dpi, bbox_inches='tight', facecolor=theme['surface'])
    print(f'wrote {outfile}')

    if args.csv is not None:
        csv_path = args.csv if isinstance(args.csv, str) else os.path.splitext(outfile)[0] + '.csv'
        pre_idx, post_idx = np.nonzero(np.isfinite(percent))
        table = np.column_stack([
            pre_idx, post_idx, n_nonzero[pre_idx, post_idx],
            n_positive[pre_idx, post_idx], percent[pre_idx, post_idx],
        ])
        np.savetxt(csv_path, table, delimiter=',', fmt=['%d', '%d', '%d', '%d', '%.4f'],
                   header='pre_block,post_block,n_nonzero,n_positive,percent_positive',
                   comments='')
        print(f'wrote {csv_path} ({len(table)} populated block pairs)')


if __name__ == '__main__':
    main()
