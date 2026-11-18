"""Predict initiation, splicing and termination profiles from sequence alone.

Chain model (initiation='chain', the default): Pol II is placed where you say (by default
at the 5' end of the region, e.g. ~100 nt upstream of a gene's TSS) in the background
state of the pre-initiation chain. One pass along the region then follows it nucleotide by
nucleotide: the walk through the promoter classes, firing (once), the early phase,
commitment to a track, splicing, the cleavage path and termination. Every track is per
polymerase placed:
  waiting               still waiting to fire at this nt
  initiation            firing here (first transcribed nt); initiation_class<c> per class
  reach                 transcribing polymerases present at this nt
  termination           terminating here (termination_cleavage: from a cleavage sub-state)
  <event>_flux          donor, acceptor, splice, cleave, commit_<T> events at this nt
  spliced, intron, cleavage, early, track_<T>   fractions of the transcribing polymerases
The model was fitted with the walk starting exactly `upstream` (100) nt before each cap,
so place Pol II the same distance upstream of the TSS you have in mind.
Or give the TSS(s) themselves (tss=): then Pol II is placed `upstream` nt before each TSS,
only the walks that first fire exactly at the TSS are kept (renormalised), and the body
starts there with their class mix: exactly how each read's body is scored in training.
The walk then only decides the class mix; nothing fires elsewhere (waiting = 0).

Legacy initiation (initiation='legacy'): TSSs from the separate initiation model (top
n_tss positions, weighted), then one profile per TSS, averaged with the weights.
"""
import numpy as np

from .genome import FastaFile, revcomp
from .sequence import SequenceStore
from .model import ReadSet, _sigmoid


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_sequence(tx, init, seq, core=None, n_tss=10, tss=None, min_spacing=1, starts=None, introns=None,
                     tss_weights=None):
    """seq: sequence oriented 5'->3' in the direction of transcription (flanks included).
    core: (c0, c1) positions of seq to report on (default: all).
    Chain model: starts = positions of seq where Pol II is placed (default: [c0], the 5' end
    of the region); several starts share the polymerases equally. Or tss = positions where
    Pol II fires (first transcribed nt; tss_weights: their shares, default equal): firing is
    conditioned on the TSS as in training (see _tss_inflow), the walk only sets the class mix.
    Legacy model: tss = TSS positions to start from instead of the predicted ones.
    introns: known introns [(first nt, last nt)] in seq coordinates (known_introns models)."""
    store = SequenceStore([seq])
    g0 = int(store.starts[0])
    c0, c1 = core or (0, len(seq))
    if tx.cfg.chain:
        out = _predict_chain(tx, store, g0, c0, c1, starts, introns=introns, tss=tss, tss_weights=tss_weights)
    else:
        out = _predict_legacy(tx, init, store, g0, c0, c1, n_tss, tss, min_spacing)
    _splice_site_scores(tx, store, g0, c0, c1, out)
    out["core"] = (c0, c1)
    return out


def _tss_inflow(tx, store, g0, c0, c1, tss, tss_weights=None):
    """Firing conditioned on the TSS, as in training: per TSS x, Pol II is placed in BG `upstream` nt
    before x (with the model's start window, if any), only the walks whose first firing is exactly at x
    are kept, and their class mix P(class | first firing at x) starts the body at x. Returns the inflow
    (N, classes) for flux_profile (weights sum to 1) and per TSS P(first firing at x) and the class mix."""
    N = c1 - c0
    tss = [int(x) for x in tss]
    w = np.ones(len(tss)) if tss_weights is None else np.asarray(tss_weights, float)
    if len(w) != len(tss) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("tss_weights: one non-negative weight per TSS, not all zero")
    w = w / w.sum()
    bad = [x for x in tss if not c0 <= x < c1]
    if bad:
        raise ValueError(f"TSS(s) {bad} outside the region [{c0}, {c1})")
    need = tx.cfg.upstream + tx.cfg.kmer_k + tx.cfg.upstream_lookahead
    if min(tss) < need:
        raise ValueError(f"TSS(s) too close to the start of the sequence: the pre-initiation walk needs "
                         f"{need} nt before each TSS (increase the flank)")
    fire = tx.initiation_scan(store, g0 + np.array(tss, np.int64))   # (n, classes)
    inflow = np.zeros((N, fire.shape[1]))
    info = []
    for x, wx, f in zip(tss, w, fire):
        p = float(f.sum())
        if p <= 0:
            raise ValueError(f"the model never fires at TSS {x} (P = 0)")
        inflow[x - c0] += wx * f / p
        info.append(dict(pos=x, weight=float(wx), p_fire_here=p, class_probs=[float(v) for v in f / p]))
    return inflow, info


def _predict_chain(tx, store, g0, c0, c1, starts=None, n_report=20, introns=None, tss=None, tss_weights=None):
    N = c1 - c0
    pos = np.arange(c0, c1)
    kin = None if introns is None else [(d - c0, a - c0) for d, a in introns]
    if tss is not None and len(tss):
        if starts:
            raise ValueError("give either Pol II start(s) or TSS(s), not both")
        inflow, tss_info = _tss_inflow(tx, store, g0, c0, c1, tss, tss_weights)
        starts = []
        fp = tx.flux_profile(store, g0 + c0, N, inflow=inflow, introns=kin)
    else:
        tss_info = None
        starts = [c0] if starts is None or len(starts) == 0 else [int(x) for x in starts]
        bad = [x for x in starts if not c0 <= x < c1]
        if bad:
            raise ValueError(f"Pol II start(s) {bad} outside the region [{c0}, {c1})")
        smass = np.zeros(N)
        for x in starts:
            smass[x - c0] += 1.0 / len(starts)
        fp = tx.flux_profile(store, g0 + c0, N, start=smass, introns=kin)
    reach = fp["reach"]
    init = fp["init"]
    out = {"waiting": fp["waiting"], "initiation": init.sum(1), "reach": reach,
           "termination": fp["term"], "termination_cleavage": fp["term_cleave"]}
    for c in range(init.shape[1]):
        out[f"initiation_class{c}"] = init[:, c]
    for k, v in fp["flux"].items():
        out[k + "_flux"] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, v in fp["occ"].items():
            out[k] = np.where(reach > 0, v / reach, np.nan)
    ini = out["initiation"]
    fired = float(ini.sum())
    order = np.argsort(-ini)[:n_report]
    out["tss"] = [dict(pos=int(pos[i]), weight=float(ini[i]),
                       class_probs=[float(x) for x in init[i] / max(init[i].sum(), 1e-300)])
                  for i in order if ini[i] > 0]
    out["tss_mass"] = float(ini[order].sum())
    out["starts"] = starts
    out["fixed_tss"] = tss_info
    out["fired_in_region"] = fired
    out["never_fired"] = fp["never_fired"]
    if fired > 0:
        cdf = np.cumsum(ini) / fired
        out["firing_quantiles"] = {q: int(pos[min(np.searchsorted(cdf, q), N - 1)]) for q in (0.1, 0.5, 0.9)}
    out["terminated_in_region"] = float(fp["term"].sum())
    return out


def _predict_legacy(tx, init, store, g0, c0, c1, n_tss, tss, min_spacing):
    N = c1 - c0
    pos = np.arange(c0, c1)
    eta = init.score(store, g0 + pos) if init is not None else np.zeros(N)
    prop = _softmax(eta)
    if tss is None:
        order = np.argsort(-prop)
        chosen = []
        for i in order:
            if pos[i] >= c1 - 2:
                continue
            if all(abs(pos[i] - pos[j]) >= min_spacing for j in chosen):
                chosen.append(i)
            if len(chosen) == n_tss:
                break
        tss_pos = pos[np.array(chosen, int)]
        wts = prop[chosen]
        mass = float(wts.sum())
        wts = wts / wts.sum()
    else:
        tss_pos = np.asarray(tss, int)
        wts = np.ones(len(tss_pos)) / len(tss_pos)
        mass = float("nan")

    rs = ReadSet.transcripts(store, g0 + tss_pos, c1 - tss_pos)
    pr = tx.profile(rs)
    lay = tx.lay
    groups = tx.state_groups(lay)
    T0 = tx.cfg.tracks[0]
    fl = ("donor", "acceptor", "splice", "cleave")
    out = {k: np.zeros(N) for k in ("reach", "termination", "termination_cleavage")}
    out.update({k: np.zeros(N) for k in groups})
    out.update({k + "_flux": np.zeros(N) for k in fl})
    per_tss = []
    cl = np.zeros(lay.S, bool); cl[lay.cl_states] = True
    for i, s in enumerate(tss_pos):
        Li = c1 - s
        sl = slice(s - c0, s - c0 + Li)
        w = wts[i]
        r = pr["reach"][i, :Li]
        occ = pr["occ"][i, :Li]
        out["reach"][sl] += w * r
        out["termination"][sl] += w * pr["term"][i, :Li]
        hz = pr["h"][i, :Li] * occ
        out["termination_cleavage"][sl] += w * r * hz[:, cl].sum(1)
        for k, msk in groups.items():
            out[k][sl] += w * r * occ[:, msk].sum(1)
        for k in fl:
            out[k + "_flux"][sl] += w * pr["flux"][k][i, :Li]
        p_first = occ[1, groups["track_" + T0]].sum() if Li > 1 else np.nan
        per_tss.append(dict(pos=int(s), weight=float(w), p_track_first=float(p_first),
                            p_terminated_in_region=float(pr["term"][i, :Li].sum())))
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in groups:
            out[k] = np.where(out["reach"] > 0, out[k] / out["reach"], np.nan)
    out["initiation"] = prop
    out["tss"] = per_tss
    out["tss_mass"] = mass
    return out


def _splice_site_scores(tx, store, g0, c0, c1, out):
    """Sequence-intrinsic splice-site strength (independent of where Pol II comes from)."""
    if not tx.cfg.splicing:
        return
    N = c1 - c0
    p = tx.p
    g = g0 + np.arange(c0, c1)
    dm = store.dinuc_mask(g, 1, tx.cfg.donor_dinucs)
    am = store.dinuc_mask(g, -1, tx.cfg.acceptor_dinucs)
    ds = np.full(N, np.nan); as_ = np.full(N, np.nan)
    ds[dm] = tx._pwm(p["u_don"], store.window_bases(g[dm], tx.don_off))
    as_[am] = tx._pwm(p["u_acc"], store.window_bases(g[am], tx.acc_off))
    # report the donor at the first intron nt (t+1), the acceptor at the last intron nt (t)
    out["donor_score"] = np.full(N, np.nan); out["donor_score"][1:] = ds[:-1]
    out["acceptor_score"] = as_
    with np.errstate(invalid="ignore"):
        for ti, T in enumerate(tx.cfg.tracks):
            out[f"donor_prob_{T}"] = _sigmoid(p["b_don"][ti] + out["donor_score"])
            out[f"acceptor_prob_{T}"] = _sigmoid(p["b_acc"][ti] + out["acceptor_score"])
    out["donor_flux"] = np.concatenate([[0.0], out["donor_flux"][:-1]])   # align to first intron nt


def predict_genomic(tx, init, fasta, chrom, start, end, strand="+", flank=1000, **kw):
    """Predict for genomic region [start, end) on one strand. Arrays come back in
    GENOMIC order (left to right), whatever the strand; TSS / start positions are genomic.
    Chain model: Pol II starts at the region's 5' end on that strand (start on '+', end-1
    on '-') unless starts=[genomic positions] is given."""
    need = tx.cfg.kmer_k + tx.cfg.upstream_lookahead + 1
    if flank < need:
        raise ValueError(f"flank must be at least {need} nt (k-mer context)")
    fa = fasta if isinstance(fasta, FastaFile) else FastaFile(fasta)
    raw = fa.fetch(chrom, start - flank, end + flank)
    seq = revcomp(raw) if strand == "-" else raw
    core = (flank, flank + (end - start))
    to_seq = (lambda g: g - (start - flank)) if strand == "+" else (lambda g: end + flank - 1 - g)
    for key in ("tss", "starts"):                        # genomic -> oriented
        if kw.get(key) is not None:
            kw[key] = [to_seq(t) for t in kw[key]]
    if kw.get("introns") is not None:                    # genomic [s, e) -> (first, last) oriented
        kw["introns"] = sorted((to_seq(s) , to_seq(e - 1)) if strand == "+" else (to_seq(e - 1), to_seq(s))
                               for s, e in kw["introns"])
    res = predict_sequence(tx, init, seq, core=core, **kw)
    if "starts" in res:
        res["starts"] = [(start - flank) + x if strand == "+" else (end + flank - 1) - x for x in res["starts"]]
    for t in res.get("fixed_tss") or []:
        t["genomic"] = (start - flank) + t["pos"] if strand == "+" else (end + flank - 1) - t["pos"]
    if "firing_quantiles" in res:
        res["firing_quantiles"] = {q: (start - flank) + x if strand == "+" else (end + flank - 1) - x
                                   for q, x in res["firing_quantiles"].items()}
    if strand == "-":
        for k, v in list(res.items()):
            if isinstance(v, np.ndarray) and v.shape == (end - start,):
                res[k] = v[::-1].copy()
    for t in res["tss"]:
        t["genomic"] = (start - flank) + t["pos"] if strand == "+" else (end + flank - 1) - t["pos"]
    res.update(chrom=chrom, start=start, end=end, strand=strand)
    return res


DEFAULT_TRACKS = ("waiting", "initiation", "reach", "termination", "termination_cleavage", "spliced",
                  "intron", "cleavage", "early", "donor_flux", "acceptor_flux", "splice_flux", "cleave_flux")


def write_bedgraphs(res, prefix, tracks=None):
    """One bedGraph per track (genomic coordinates, from predict_genomic). Default: the
    tracks above plus the fraction of polymerases on each track and per-class initiation."""
    if tracks is None:
        tracks = list(DEFAULT_TRACKS) + sorted(k for k in res if k.startswith(("track_", "initiation_class")))
    paths = []
    for k in tracks:
        if k not in res:
            continue
        v = res[k]
        path = f"{prefix}.{res['strand'].replace('+', 'plus').replace('-', 'minus')}.{k}.bedGraph"
        with open(path, "w") as fh:
            fh.write(f"track type=bedGraph name={k}_{res['strand']}\n")
            for i, x in enumerate(v):
                if np.isfinite(x) and x != 0:
                    fh.write(f"{res['chrom']}\t{res['start'] + i}\t{res['start'] + i + 1}\t{x:.6g}\n")
        paths.append(path)
    return paths
