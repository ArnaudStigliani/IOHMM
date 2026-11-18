"""Known intron structures from read junctions (no annotation).

  python -m txhmm structures --source big.npz --data train.npz --out train_introns.npz \
      [--min-junction-reads 2] [--min-intron 30]

1. From the SOURCE dataset (can be larger than the training one: a structure is a property of
   the genomic locus, not of the reads used for fitting): every canonical junction of every
   read (all junctions, not only the first) is counted once per read, in genomic coordinates
   on the read's strand.
2. Junctions with >= min_junction_reads reads and intron length >= min_intron are kept.
   Overlapping introns on the same strand compete: the one with more supporting reads wins
   (ties: the shorter, then the leftmost). Unspliced reads never remove an intron.
3. The surviving introns are mapped onto the loci of the TARGET dataset (introns lying fully
   inside a locus, oriented locus coordinates) and saved as `kin_ptr`, `kin_don`, `kin_acc`
   (first and last intron nt). Loci without any get an empty list ("no intron").
"""
import bisect
from collections import defaultdict
import numpy as np

from .dataprep import TxDataset


def junction_counts(ds, canonical_only=True):
    """{(chrom, strand, gfirst, glast): reads} in genomic coordinates (gfirst = first intron
    nt in the direction of transcription)."""
    r, L = ds.reads, ds.loci
    ptr = r["jptr"]
    g0, g1 = np.asarray(L["g0"], np.int64), np.asarray(L["g1"], np.int64)
    counts = defaultdict(int)
    for i in np.flatnonzero(ptr[1:] > ptr[:-1]):
        l = int(r["locus"][i]); ch, st = str(L["chrom"][l]), str(L["strand"][l])
        seen = set()
        for k in range(ptr[i], ptr[i + 1]):
            if canonical_only and not ds.jcanon[k]:
                continue
            d, a = int(ds.jdon[k]), int(ds.jacc[k])
            gd, ga = (g0[l] + d, g0[l] + a) if st == "+" else (g1[l] - 1 - d, g1[l] - 1 - a)
            key = (ch, st, int(gd), int(ga))
            if key not in seen:
                seen.add(key); counts[key] += 1
    return counts


def resolve(counts, min_reads=2, min_intron=30):
    """Non-overlapping introns per (chrom, strand): {(chrom, strand): sorted [(lo, hi, gfirst, glast, n)]}."""
    cand = defaultdict(list)
    for (ch, st, gd, ga), n in counts.items():
        lo, hi = min(gd, ga), max(gd, ga)
        if n >= min_reads and hi - lo + 1 >= min_intron:
            cand[(ch, st)].append((n, lo, hi, gd, ga))
    out = {}
    for key, v in cand.items():
        v.sort(key=lambda x: (-x[0], x[2] - x[1], x[1]))
        los, his, kept = [], [], []
        for n, lo, hi, gd, ga in v:
            j = bisect.bisect_left(los, lo)
            if j > 0 and his[j - 1] >= lo:            # overlaps the kept intron to the left
                continue
            if j < len(los) and los[j] <= hi:         # overlaps the kept intron to the right
                continue
            los.insert(j, lo); his.insert(j, hi); kept.insert(j, (lo, hi, gd, ga, n))
        out[key] = kept
    return out


def attach(ds, introns):
    """kin dict (ptr, don, acc) for the loci of ds."""
    L = ds.loci
    nl = len(L["chrom"])
    g0, g1 = np.asarray(L["g0"], np.int64), np.asarray(L["g1"], np.int64)
    arr = {k: np.array([x[:2] for x in v], np.int64).reshape(-1, 2) for k, v in introns.items()}
    don, acc, ptr = [], [], [0]
    for l in range(nl):
        st = str(L["strand"][l])
        a_ = arr.get((str(L["chrom"][l]), st))
        if a_ is not None and len(a_):
            j0 = np.searchsorted(a_[:, 0], g0[l], "left")
            j1 = np.searchsorted(a_[:, 0], g1[l], "left")
            sel = a_[j0:j1]
            sel = sel[sel[:, 1] < g1[l]]
            if st == "+":
                d, a = sel[:, 0] - g0[l], sel[:, 1] - g0[l]
            else:
                d, a = g1[l] - 1 - sel[:, 1], g1[l] - 1 - sel[:, 0]
            o = np.argsort(d)
            don.extend(d[o].tolist()); acc.extend(a[o].tolist())
        ptr.append(len(don))
    return dict(ptr=np.array(ptr, np.int64), don=np.array(don, np.int64), acc=np.array(acc, np.int64))


def region_introns(data_path, chrom, start, end, strand=None):
    """Known introns of a dataset (txhmm structures) lying fully inside chrom:[start, end), as genomic
    0-based half-open [(start, end, strand)], sorted and without duplicates. Reads only the locus and
    kin_* arrays of the .npz (not the sequence)."""
    z = np.load(data_path, allow_pickle=False)
    if "kin_ptr" not in z.files:
        raise SystemExit(f"{data_path} has no known introns (make it with `txhmm structures`)")
    ch, sd = z["locus_chrom"], z["locus_strand"]
    g0, g1 = z["locus_g0"].astype(np.int64), z["locus_g1"].astype(np.int64)
    ptr, don, acc = z["kin_ptr"], z["kin_don"], z["kin_acc"]
    out = set()
    for l in np.flatnonzero((ch == chrom) & (g1 > start) & (g0 < end)):
        st = str(sd[l])
        if strand is not None and st != strand:
            continue
        for d, a in zip(don[ptr[l]:ptr[l + 1]], acc[ptr[l]:ptr[l + 1]]):
            s, e = (g0[l] + d, g0[l] + a + 1) if st == "+" else (g1[l] - 1 - a, g1[l] - d)
            if s >= start and e <= end:
                out.add((int(s), int(e), st))
    return sorted(out)


def region_caps(data_path, chrom, start, end, strand):
    """Caps of a dataset's reads lying inside chrom:[start, end) on one strand: genomic 0-based
    positions and their read counts, sorted by position. Reads only the locus and read_locus /
    read_tss arrays of the .npz (not the sequence)."""
    z = np.load(data_path, allow_pickle=False)
    ch, sd = z["locus_chrom"], z["locus_strand"]
    g0, g1 = z["locus_g0"].astype(np.int64), z["locus_g1"].astype(np.int64)
    loci = np.flatnonzero((ch == chrom) & (sd == strand) & (g1 > start) & (g0 < end))
    rl, rt = z["read_locus"], z["read_tss"].astype(np.int64)
    m = np.isin(rl, loci)
    l, t = rl[m], rt[m]
    g = g0[l] + t if strand == "+" else g1[l] - 1 - t
    g = g[(g >= start) & (g < end)]
    pos, n = np.unique(g, return_counts=True)
    return pos, n


def build(source_path, data_path, out_path, min_reads=2, min_intron=30, verbose=True):
    src = TxDataset.load(source_path)
    counts = junction_counts(src)
    introns = resolve(counts, min_reads, min_intron)
    n_kept = sum(len(v) for v in introns.values())
    n_cand = sum(1 for n in counts.values() if n >= min_reads)
    del src
    ds = TxDataset.load(data_path)
    ds.kin = attach(ds, introns)
    ds.meta = dict(ds.meta, structures=dict(source=str(source_path), min_junction_reads=min_reads,
                                            min_intron=min_intron, introns_genome=n_kept))
    ds.save(out_path)
    if verbose:
        nper = np.diff(ds.kin["ptr"])
        used = np.unique(ds.reads["locus"])
        print(f"[structures] junctions in source: {len(counts):,} (>= {min_reads} reads: {n_cand:,}); "
              f"kept after length and overlap filters: {n_kept:,}")
        print(f"[structures] training loci with reads: {len(used):,}; with >= 1 known intron: "
              f"{int((nper[used] > 0).sum()):,}; introns attached: {int(nper[used].sum()):,}; saved {out_path}")
    return ds
