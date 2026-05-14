"""Batch-local hidden-cell coalescing analysis.

For each batch (BATCH=8 events) flowing through the SSLA-S pipeline, count:
  - touches: # of active (warp, event) pairs (active = event passes warp's mask
             and target cell is in-bounds)
  - unique_cells: # of distinct (py, px) target cells touched by this warp
                  in this batch
  - reuse = touches / unique_cells

If reuse > 1.0, some cells are hit multiple times within one batch by the same
warp → coalescing hidden-state load/store across those events could help.
If reuse ≈ 1.0, coalescing saves nothing.

NO kernel changes. Pure-Python simulation of the cell-owner mapping logic,
tdrop_s2/s3 counters, and pool. Uses same RNG seed as perf_celled_multi.py
for reproducibility.
"""
import argparse
import numpy as np

H2, W2 = 16, 20
H3, W3 = H2 >> 1, W2 >> 1
BATCH = 8
N_WARPS = 9


def warp_targets(evx, evy, mask, warp, H, W):
    """For one warp over a batch of BATCH events, compute touches and uniques.

    cell-owner mapping (mirroring run_layer_celled_implicit):
      dy = ((own_y - evy) % 3 + 3) % 3 - 1
      dx = ((own_x - evx) % 3 + 3) % 3 - 1
      target cell = (evy + dy, evx + dx); active if mask[e] != 0 AND in-bounds.
    """
    own_y = warp // 3
    own_x = warp % 3
    dy = ((own_y - evy) % 3 + 3) % 3 - 1
    dx = ((own_x - evx) % 3 + 3) % 3 - 1
    py = evy + dy
    px = evx + dx
    active = mask.astype(bool) & (py >= 0) & (py < H) & (px >= 0) & (px < W)
    if not active.any():
        return 0, 0
    pairs = py[active].astype(np.int64) * W + px[active].astype(np.int64)
    return int(active.sum()), int(len(np.unique(pairs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200_000,
                    help='Total events to simulate (matches perf_celled_multi default)')
    ap.add_argument('--tdrop', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    evx_all = rng.integers(0, W2, args.n).astype(np.int32)
    evy_all = rng.integers(0, H2, args.n).astype(np.int32)

    # Per-cell tdrop counters (modular).
    tdrop_s2 = np.zeros(H2 * W2, dtype=np.int64)
    tdrop_s3 = np.zeros(H3 * W3, dtype=np.int64)

    layers = ['L4', 'L5', 'L6', 'L7']
    touches_total = {ln: np.zeros(N_WARPS, dtype=np.int64) for ln in layers}
    unique_total  = {ln: np.zeros(N_WARPS, dtype=np.int64) for ln in layers}
    # Also per-cell-count histogram: how often does a warp see k events on
    # the same cell in one batch?
    cell_count_hist = {ln: np.zeros(BATCH + 1, dtype=np.int64) for ln in layers}
    n_batches = 0
    pass2_total = 0

    for b in range(0, args.n - BATCH + 1, BATCH):
        evx = evx_all[b:b+BATCH]
        evy = evy_all[b:b+BATCH]

        # ---- tdrop_s2 ----
        pass2 = np.zeros(BATCH, dtype=np.int32)
        for e in range(BATCH):
            idx = evy[e] * W2 + evx[e]
            pre = tdrop_s2[idx]
            tdrop_s2[idx] = pre + 1
            pass2[e] = 1 if ((pre % args.tdrop) == 0) else 0
        pass2_total += int(pass2.sum())

        # ---- L4 / L5 (s2 grid, all events) ----
        all_mask = np.ones(BATCH, dtype=np.int32)
        for warp in range(N_WARPS):
            t, u = warp_targets(evx, evy, all_mask, warp, H2, W2)
            touches_total['L4'][warp] += t
            unique_total['L4'][warp]  += u
            touches_total['L5'][warp] += t
            unique_total['L5'][warp]  += u
            # Histogram: count events per cell for this warp
            if t > 0:
                own_y = warp // 3
                own_x = warp % 3
                dy = ((own_y - evy) % 3 + 3) % 3 - 1
                dx = ((own_x - evx) % 3 + 3) % 3 - 1
                py = evy + dy
                px = evx + dx
                active = (py >= 0) & (py < H2) & (px >= 0) & (px < W2)
                pairs = (py[active] * W2 + px[active]).astype(np.int64)
                _, counts = np.unique(pairs, return_counts=True)
                for c in counts:
                    cell_count_hist['L4'][min(c, BATCH)] += 1
                    cell_count_hist['L5'][min(c, BATCH)] += 1

        # ---- pool s2 → s3 ----
        evx_s3 = evx >> 1
        evy_s3 = evy >> 1

        # ---- tdrop_s3 (gated on pass2) ----
        pass3 = np.zeros(BATCH, dtype=np.int32)
        for e in range(BATCH):
            if pass2[e] == 0:
                continue
            idx = evy_s3[e] * W3 + evx_s3[e]
            pre = tdrop_s3[idx]
            tdrop_s3[idx] = pre + 1
            pass3[e] = 1 if ((pre % args.tdrop) == 0) else 0

        # ---- L6 / L7 (s3 grid, pass2 events) ----
        for warp in range(N_WARPS):
            t, u = warp_targets(evx_s3, evy_s3, pass2, warp, H3, W3)
            touches_total['L6'][warp] += t
            unique_total['L6'][warp]  += u
            touches_total['L7'][warp] += t
            unique_total['L7'][warp]  += u
            if t > 0:
                own_y = warp // 3
                own_x = warp % 3
                dy = ((own_y - evy_s3) % 3 + 3) % 3 - 1
                dx = ((own_x - evx_s3) % 3 + 3) % 3 - 1
                py = evy_s3 + dy
                px = evx_s3 + dx
                active = pass2.astype(bool) & (py >= 0) & (py < H3) & (px >= 0) & (px < W3)
                pairs = (py[active] * W3 + px[active]).astype(np.int64)
                _, counts = np.unique(pairs, return_counts=True)
                for c in counts:
                    cell_count_hist['L6'][min(c, BATCH)] += 1
                    cell_count_hist['L7'][min(c, BATCH)] += 1

        n_batches += 1

    # ---- Report ----
    print(f"\nAnalyzed {n_batches} batches ({n_batches * BATCH} events)")
    print(f"pass2 rate: {pass2_total / (n_batches * BATCH):.3f} "
          f"(expected ≈ 1/{args.tdrop})\n")

    print(f"{'layer':<5} {'warp':<5} {'touches':>10} {'unique':>10} "
          f"{'reuse':>8} {'savings':>10}")
    print("-" * 60)
    for ln in layers:
        for warp in range(N_WARPS):
            t = touches_total[ln][warp] / n_batches
            u = unique_total[ln][warp] / n_batches
            r = t / u if u > 0 else 0
            # "savings" = fraction of hidden state load/store skipped if we coalesced
            sav = (1 - 1/r) * 100 if r > 0 else 0
            print(f"{ln:<5} {warp:<5} {t:>10.3f} {u:>10.3f} "
                  f"{r:>8.3f} {sav:>9.1f}%")
        t_all = touches_total[ln].sum() / n_batches
        u_all = unique_total[ln].sum() / n_batches
        r_all = t_all / u_all if u_all > 0 else 0
        sav_all = (1 - 1/r_all) * 100 if r_all > 0 else 0
        print(f"{ln:<5} {'sum':<5} {t_all:>10.3f} {u_all:>10.3f} "
              f"{r_all:>8.3f} {sav_all:>9.1f}%")
        print()

    print(f"\nPer-cell event-count histogram (how many events hit one cell "
          f"in one batch, summed across all warps and batches):")
    print(f"{'layer':<5} {'k=1':>10} {'k=2':>10} {'k=3':>10} {'k=4':>10} "
          f"{'k=5+':>10}")
    for ln in layers:
        h = cell_count_hist[ln]
        print(f"{ln:<5} {h[1]:>10} {h[2]:>10} {h[3]:>10} {h[4]:>10} "
              f"{h[5:].sum():>10}")


if __name__ == '__main__':
    main()
