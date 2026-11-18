"""Read out what a fitted model learned, as tables.

* termination k-mers ('random' route): effect of the k-mer ending at the polymerase
  position (the last k nt transcribed) on the per-nt termination hazard of each track.
* transitions: every k-mer-driven transition of the model, one table each, with the
  k-mer's effect (log-odds), the transition probability without and with the k-mer
  (softmax over all the transitions of the source state at that position) and, with a
  dataset, the support: expected number of these transitions inferred in the reads
  (posterior), the exposure (expected time in the source state) and their ratio.
    chain model: penter / padv / pexit / initiate (per promoter class; k-mer ending
    `upstream_lookahead` nt downstream of the position), commit (per class and track;
    k-mer just transcribed), cleave (entry into the cleavage path, per track).
* distributions implied by the parameters: cleavage delay (entry -> termination), early
  phase (time to commit, track), pre-initiation sub-state transitions.
* posterior position profiles (with a dataset): where, relative to the cap, the promoter
  classes are entered, where polymerases commit, and how far upstream of the read 3' end
  the cleavage path was entered.
* PWMs: 5' and 3' splice sites (and the legacy initiation model).
"""
import numpy as np

from .sequence import kmer_names

BASES = "ACGT"


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================ termination k-mers

def kmer_table(tx, ds=None, max_positions=50_000_000, seed=0):
    """One row per k-mer: effect on the random-route termination hazard per k-mer group,
    the hazard per nt with that k-mer (base state of each track), and (with a dataset)
    occurrences along reads, at read 3' ends, and the raw end enrichment."""
    if not tx.cfg.kmer_k:
        raise ValueError("this model has no k-mer block (kmer_k = 0)")
    k = tx.cfg.kmer_k
    names = kmer_names(k)[:-1]                       # drop 'UNK'
    groups = list(tx.cfg.tracks) if tx.nkg == len(tx.cfg.tracks) else ["all"]
    rows = {"kmer": names}
    for j, gname in enumerate(groups):
        rows[f"effect_{gname}"] = tx.p["vk"][j, :4 ** k]
    for ti, T in enumerate(tx.cfg.tracks):
        j = 0 if tx.nkg == 1 else ti
        rows[f"hazard_per_nt_{T}"] = _sig(tx.p["v"][ti, 0] + tx.p["vk"][j, :4 ** k])
    if ds is not None:
        L = ds.lengths.astype(np.int64)
        gstart = ds.gstart()
        idx = np.arange(len(L))
        if L.sum() > max_positions:                  # subsample reads for very large datasets
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(L))
            idx = idx[np.cumsum(L[idx]) <= max_positions]
        Ls = L[idx]
        pos = np.repeat(gstart[idx], Ls) + (np.arange(Ls.sum()) - np.repeat(np.cumsum(Ls) - Ls, Ls))
        kid = ds.store.kmer_index(pos, k)
        n_pos = np.bincount(kid, minlength=4 ** k + 1)[:4 ** k]
        ends = gstart[idx] + Ls - 1
        n_end = np.bincount(ds.store.kmer_index(ends, k), minlength=4 ** k + 1)[:4 ** k]
        rows["n_positions"] = n_pos
        rows["n_read_ends"] = n_end
        with np.errstate(divide="ignore", invalid="ignore"):
            rows["end_enrichment"] = (n_end / max(n_end.sum(), 1)) / (n_pos / max(n_pos.sum(), 1))
    return rows


def top_kmers(tx, n=20, ds=None, group=0, direction="up"):
    """The n k-mers that raise ('up') or lower ('down') random-route termination the most."""
    t = kmer_table(tx, ds)
    col = [c for c in t if c.startswith("effect_")][group]
    order = np.argsort(-t[col] if direction == "up" else t[col])[:n]
    return [{c: (t[c][i] if c == "kmer" else float(t[c][i])) for c in t} for i in order]


# ============================================================ PWMs

def pwm_table(u, offsets, anchor_name):
    """PWM as rows (offset, A, C, G, T) in log-odds, centred per position (mean 0)."""
    w = u[:, :4] - u[:, :4].mean(1, keepdims=True)
    return {"offset": np.asarray(offsets), **{b: w[:, i] for i, b in enumerate(BASES)},
            "anchor": anchor_name}


def consensus(table, threshold=0.3):
    """Upper case: clear preference (>= threshold above the mean); lower case: weak; n: none."""
    out = []
    for i in range(len(table["offset"])):
        v = np.array([table[b][i] for b in BASES])
        j = int(np.argmax(v))
        out.append(BASES[j] if v[j] >= threshold else (BASES[j].lower() if v[j] > 0.1 else "n"))
    return "".join(out)


def model_pwms(tx, init=None):
    """Donor / acceptor (/ legacy initiation) PWMs with offsets relative to the site."""
    out = {}
    if tx.cfg.splicing:
        # donor window is relative to t, with the first intron nt at t+1
        out["donor"] = pwm_table(tx.p["u_don"], tx.don_off - 1,
                                 "offset 0 = first intron nt (the G of GT)")
        out["acceptor"] = pwm_table(tx.p["u_acc"], tx.acc_off,
                                    "offset 0 = last intron nt (the G of AG)")
    if init is not None:
        out["initiation"] = pwm_table(init.u, init.off, "offset 0 = TSS (first transcribed nt)")
    return out


def write_tsv(path, table):
    cols = [c for c in table if c != "anchor"]
    n = len(table[cols[0]])
    with open(path, "w") as fh:
        if "anchor" in table:
            fh.write(f"# {table['anchor']}\n")
        fh.write("\t".join(cols) + "\n")
        for i in range(n):
            vals = []
            for c in cols:
                v = table[c][i]
                vals.append(v if isinstance(v, str) else (f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)))
            fh.write("\t".join(vals) + "\n")


# ============================================================ transition groups

def transition_groups(tx):
    """Every k-mer-driven transition of the model, as groups of edges sharing one k-mer
    table: name -> dict(kind, cls, track, table (array over k-mers), ref_state (name of
    the state whose softmax gives the probabilities), source ('cur'|'la'), where (text))."""
    c, p, lay = tx.cfg, tx.p, tx.lay
    K = 4 ** c.kmer_k if c.kmer_k else 0
    la = c.upstream_lookahead
    out = {}
    if not K:
        return out
    where_la = (f"k-mer ending {la} nt downstream of the position u where the transition is "
                f"taken (covers u{la - c.kmer_k + 1:+d}..u{la:+d})")
    where_cur = "k-mer just transcribed (ending at the polymerase position t)"
    if c.chain:
        for ci in range(c.promoter_classes):
            out[f"penter_class{ci}"] = dict(kind="penter", cls=ci, track=-1, table=p["vk_penter"][ci, :K],
                                            ref_state="BG", source="la",
                                            where=where_la + "; background -> first sub-state of the class")
            out[f"padv_class{ci}"] = dict(kind="padv", cls=ci, track=-1, table=p["vk_padv"][ci, :K],
                                          ref_state=f"C{ci}.0", source="la",
                                          where=where_la + "; advance to the next promoter sub-state")
            out[f"pexit_class{ci}"] = dict(kind="pexit", cls=ci, track=-1, table=p["vk_pexit"][ci, :K],
                                           ref_state=f"C{ci}.0", source="la",
                                           where=where_la + "; back to background")
            jbest = int(np.argmax(p["b_init"][ci]))
            out[f"initiate_class{ci}"] = dict(
                kind="initiate", cls=ci, track=-1, table=p["vk_init"][ci, :K], ref_state=f"C{ci}.{jbest}",
                source="la", where=(f"k-mer covering TSS{la - c.kmer_k:+d}..TSS{la - 1:+d} (u = the nt "
                                    f"before the TSS); probabilities at sub-state {jbest}, the one "
                                    f"with the highest initiation bias"))
            for ti, T in enumerate(c.tracks):
                out[f"commit_class{ci}_to_{T}"] = dict(kind="ecommit", cls=ci, track=ti,
                                                       table=p["vk_ecommit"][ci, ti, :K],
                                                       ref_state=f"E{ci}.0", source="cur",
                                                       where=where_cur + "; early phase -> track")
    if c.cleave_substates:
        for ti, T in enumerate(c.tracks):
            out[f"cleave_{T}"] = dict(kind="cleave", cls=-1, track=ti, table=p["vk_cleave"][ti, :K],
                                      ref_state=T, source="cur",
                                      where=where_cur + "; entry into the cleavage path "
                                                        "(probabilities from the unspliced base state)")
    for ki, kind in enumerate(c.kmer_transitions):
        for ti, T in enumerate(c.tracks):
            src = {"commit": "INIT", "donor": T, "acceptor": T + "ps", "abandon": T + "ps",
                   "splice": f"{T}w0", "switch": T}[kind]
            if src not in lay.idx:
                continue
            out[f"{kind}_{T}"] = dict(kind=kind, cls=-1, track=ti, table=p["vkt"][ki, ti, :K],
                                      ref_state=src, source="cur", where=where_cur)
    return out


def _group_edges(lay, g):
    k = g["kind"]
    if k not in lay.edges_of:
        return np.array([], int)
    e = lay.edges_of[k]
    if g["cls"] >= 0:
        e = e[lay.ecls[e] == g["cls"]]
    if g["track"] >= 0:
        e = e[lay.etrack[e] == g["track"]]
    return e


def _state_probs(tx, lay, s, kmer_ids, with_kmers=True):
    """Probabilities of the outgoing transitions of state s, for each k-mer (n, n_out),
    at a position that is not a splice site (with_kmers=False: biases only); edge ids."""
    p = tx.p
    out = lay.out[s]
    Z = np.zeros((len(kmer_ids), len(out)))
    for j, e in enumerate(out):
        k = lay.kind[e]
        if k in ("donor", "acceptor"):
            Z[:, j] = -np.inf
        elif k == "commit":
            Z[:, j] = p["w_commit"][0]
        else:
            sp = tx._edge_spec(lay, e)
            if sp is not None:
                bn, bi, tn, ti, _ = sp
                Z[:, j] = p[bn][bi] + (p[tn][ti][kmer_ids] if (tn is not None and with_kmers) else 0.0)
            ex = tx._edge_extra(lay, e)
            if ex is not None:
                Z[:, j] += p[ex[0]][ex[1]]
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True), out


def transition_table(tx, name, g, support=None):
    """Table over all k-mers for one transition group."""
    lay = tx.lay
    K = 4 ** tx.cfg.kmer_k
    names = kmer_names(tx.cfg.kmer_k)[:-1]
    s = lay.idx[g["ref_state"]]
    edges = set(_group_edges(lay, g).tolist())
    P, out = _state_probs(tx, lay, s, np.arange(K))
    P0, _ = _state_probs(tx, lay, s, np.zeros(1, int), with_kmers=False)
    cols = [j for j, e in enumerate(out) if e in edges]
    tab = {"kmer": names, "effect": np.asarray(g["table"], float),
           "prob_base": np.full(K, P0[0, cols].sum()), "prob_with_kmer": P[:, cols].sum(1)}
    if support is not None and name in support["trans"]:
        tr, ex = support["trans"][name][:K], support["expo"][name][:K]
        tab["expected_transitions"] = tr
        tab["exposure"] = ex
        with np.errstate(invalid="ignore", divide="ignore"):
            tab["observed_rate"] = np.where(ex > 0, tr / ex, np.nan)
    return tab


# ============================================================ implied distributions

def cleavage_delay(tx, track=0, max_nt=2000):
    """Distance (nt) from entry into the cleavage path to termination implied by the
    sub-state hazards ('cut') and advance rate: pmf over 1..max_nt and a summary."""
    p, n = tx.p, tx.cfg.cleave_substates
    h = _sig(p["cut"][track])
    a = _sig(p["b_cleave_adv"][track]) if n > 1 else 0.0
    occ = np.zeros(n); occ[0] = 1.0                              # in sub-state 0 one nt after entry
    pmf = np.zeros(max_nt)
    for d in range(max_nt):
        term = occ * h
        pmf[d] = term.sum()
        surv = occ - term
        nxt = surv * (1 - a)
        if n > 1:
            nxt[1:] += surv[:-1] * a
            nxt[-1] += surv[-1] * a                              # the last sub-state has no exit
        occ = nxt
    d = np.arange(1, max_nt + 1)
    cdf = np.cumsum(pmf)
    q = lambda x: int(d[min(np.searchsorted(cdf, x * cdf[-1]), max_nt - 1)])
    return pmf, {"terminated_within_max": float(cdf[-1]),
                 "mean": float((pmf * d).sum() / max(cdf[-1], 1e-300)),
                 "median": q(0.5), "p10": q(0.1), "p90": q(0.9),
                 "within_50nt": float(cdf[min(49, max_nt - 1)]),
                 "cut_per_substate": h.tolist(), "advance_per_nt": float(a)}


def early_phase(tx, cls=0, max_nt=500):
    """Early phase of a promoter class at baseline (no k-mer effects): pmf of the commit
    position (nt after the TSS) and P(track)."""
    c, p = tx.cfg, tx.p
    en, ntr = c.early_substates, len(c.tracks)
    adv = p["b_eadv"][cls] if en > 1 else -np.inf
    occ = np.zeros(en); occ[0] = 1.0
    pmf = np.zeros(max_nt); ptrack = np.zeros(ntr)
    for d in range(max_nt):
        nxt = np.zeros(en)
        for j in range(en):
            z = np.concatenate([[0.0], [adv if j + 1 < en else -np.inf], p["b_ecommit"][cls]])
            pr = np.exp(z - z.max()); pr /= pr.sum()
            nxt[j] += occ[j] * pr[0]
            if j + 1 < en:
                nxt[j + 1] += occ[j] * pr[1]
            ptrack += occ[j] * pr[2:]
            pmf[d] += occ[j] * pr[2:].sum()
        occ = nxt
    dd = np.arange(1, max_nt + 1)
    tot = max(pmf.sum(), 1e-300)
    cdf = np.cumsum(pmf) / tot
    return pmf, {"P(track)": dict(zip(c.tracks, (ptrack / max(ptrack.sum(), 1e-300)).round(4).tolist())),
                 "mean_commit_nt_after_tss": float((pmf * dd).sum() / tot),
                 "median": int(dd[np.searchsorted(cdf, 0.5)]),
                 "committed_within_max": float(pmf.sum())}


def promoter_chain_table(tx):
    """Baseline (no k-mer) transition probabilities of every promoter sub-state."""
    c, p, lay = tx.cfg, tx.p, tx.lay
    rows = {"state": [], "stay": [], "advance": [], "exit": [], "initiate": []}
    for ci in range(c.promoter_classes):
        for j in range(c.promoter_substates):
            s = lay.idx[f"C{ci}.{j}"]
            zs, kinds = [], []
            for e in lay.out[s]:
                k = lay.kind[e]
                sp = tx._edge_spec(lay, e)
                zs.append(0.0 if sp is None else p[sp[0]][sp[1]])
                kinds.append(k)
            z = np.array(zs); pr = np.exp(z - z.max()); pr /= pr.sum()
            d = dict(zip(kinds, pr))
            rows["state"].append(f"C{ci}.{j}")
            rows["stay"].append(d.get("ref", 0.0)); rows["advance"].append(d.get("padv", 0.0))
            rows["exit"].append(d.get("pexit", 0.0)); rows["initiate"].append(d.get("initiate", 0.0))
    return {k: (np.array(v) if k != "state" else v) for k, v in rows.items()}


# ============================================================ posterior support

def _posteriors(tx, bt, init=None, terminal=None):
    """Posteriors of one batch (fitting layouts, delay line included): gamma (B,T,S),
    gamma_h (incl. young-intron mass, for hazards), xi_d (B,T-1,Ed) for the position-
    dependent transitions, Pd, h, r0 ..."""
    bt.want_xi = True
    bt.init, bt.terminal = init, terminal
    fb = tx.forward_backward(bt)
    for k in ("gamma", "gamma_h", "xi_d", "xi_young", "gamma_young"):
        if k in fb:
            fb[k][fb["dead"]] = 0.0
    return fb


def transition_support(tx, ds, n_reads=2000, seed=0, max_pos=3000, max_tokens=100_000):
    """Forward-backward on up to n_reads reads: per transition group and k-mer, the expected
    number of transitions and the exposure (expected time in a source state where the
    transition was possible); position profiles of class occupancy / class entry /
    commitment relative to the cap and of cleavage entry relative to the read 3' end;
    expected terminations at read ends by route, track and end k-mer; posterior
    class x track counts."""
    from .model import ReadSet, TERM, MIX
    c = tx.cfg
    rs = ReadSet.from_dataset(ds)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rs), min(n_reads, len(rs)), replace=False))
    sub = rs.subset(idx)
    K1 = tx.K1 if c.kmer_k else 1
    W = c.upstream if c.chain else 0
    groups = transition_groups(tx)
    trans = {n: np.zeros(K1) for n in groups}
    expo = {n: np.zeros(K1) for n in groups}
    prof = {}                                         # name -> histogram; index i <-> position i - W
    ntr = len(c.tracks)
    term_random = np.zeros((ntr, K1)); term_cleave = np.zeros((ntr, K1))
    class_track = np.zeros((max(c.promoter_classes, 1), ntr))
    cleave_state = {}                                 # track -> splicing state -> [entries, exposure]
    n_used = 0

    def hist(name, offset, values, size):
        h = prof.setdefault(name, np.zeros(size))
        pos = np.arange(values.shape[1]) + offset
        ok = (pos >= 0) & (pos < size)
        h[pos[ok]] += values.sum(0)[ok]

    def accumulate(post, bt, lay, kid, part):
        xi, gam, Pd = post["xi_d"], post["gamma"], post["Pd"]
        pd = lay.pos_d
        for n, g in groups.items():
            if (g["source"] == "la") != (part == "prefix"):
                continue
            es = _group_edges(lay, g)
            if not len(es):
                continue
            x = xi[:, :, pd[es]].sum(2)
            ex = (gam[:, :-1][:, :, lay.src[es]] * (Pd[:, :-1][:, :, pd[es]] > 0)).sum(2) * bt.tmask
            trans[n] += np.bincount(kid.ravel(), x.ravel(), minlength=K1)
            expo[n] += np.bincount(kid.ravel(), ex.ravel(), minlength=K1)
            if g["kind"] in ("penter", "ecommit"):
                # position of the state entered, relative to the cap
                hist(n, 1 if part == "prefix" else 1 + W, x, W + max_pos)
        if part == "prefix":
            for ci in range(c.promoter_classes):     # where each class sits before the cap
                st = [lay.idx[f"C{ci}.{j}"] for j in range(c.promoter_substates)]
                hist(f"occupancy_class{ci}", 0, gam[:, :, st].sum(2), W + max_pos)

    for b in tx._buckets(sub.L, 32, max_tokens=max_tokens):
        if c.chain:
            pre, body = tx.lays_fit["prefix"], tx.lays_fit["body"]
            bp = tx.make_batch(sub, b, "prefix")
            fp = tx.forward_backward(bp, posteriors=False)
            br = tx.make_batch(sub, b, "read")
            init = np.zeros((br.B, body.S)); init[:, body.e0_states] = fp["a_last"][:, pre.e0_states]
            pb = _posteriors(tx, br, init=init)
            terminal = np.zeros((bp.B, pre.S)); terminal[:, pre.e0_states] = pb["r0"][:, body.e0_states]
            pp = _posteriors(tx, bp, terminal=terminal)
            dead = fp["dead"] | pb["dead"] | pp["dead"]
            for d in (pb, pp):
                for k in ("gamma", "gamma_h", "xi_d"):
                    d[k][dead] = 0.0
            accumulate(pp, bp, pre, bp.kidx_la[:, :-1], "prefix")
            accumulate(pb, br, body, br.kidx[:, :-1], "body")
            for e in body.edges_of["ecommit"]:
                class_track[body.ecls[e], body.etrack[e]] += pb["xi_d"][:, :, body.pos_d[e]].sum()
            post, bt, lay = pb, br, body
        else:
            bt = tx.make_batch(sub, b, "read")
            lay = tx.lays_fit["full"]
            post = _posteriors(tx, bt)
            accumulate(post, bt, lay, bt.kidx[:, :-1] if c.kmer_k else np.zeros((bt.B, bt.T - 1), int), "body")
            dead = post["dead"]
        n_used += int((~dead).sum())
        rows = np.arange(bt.B)
        Lm1 = bt.L - 1
        # cleavage entries by the splicing state they come from (young intron: delay line)
        if c.cleave_substates and bt.T > 1:
            kidc = bt.kidx[:, :-1]
            for ti, T in enumerate(c.tracks):
                d = cleave_state.setdefault(T, {k: [0.0, 0.0] for k in CLEAVE_STATES})
                es = lay.edges_of["cleave"]
                for e in es[lay.etrack[es] == ti]:
                    sname = lay.states[lay.src[e]]["name"]
                    sb = lay.sub[lay.src[e]]
                    if sb == "ps" and sname != T + "ps":
                        g = "young intron"
                    else:
                        g = {"base": "unspliced", "ps": "in intron", "w": "waiting", "s": "spliced",
                             "ps2": "later intron", "w2": "waiting"}[sb]
                    d[g][0] += float(post["xi_d"][:, :, lay.pos_d[e]].sum())
                    d[g][1] += float((post["gamma"][:, :-1, lay.src[e]] * bt.tmask).sum())
                if "xi_young" in post:                  # delay lanes of this track
                    for ln in np.flatnonzero(lay.lane_track == ti):
                        xy, gy = post["xi_young"][:, :, ln], post["gamma_young"][:, :-1, ln] * bt.tmask
                        k = "young intron" if lay.lane_row[ln] == 0 else "later intron"
                        d[k][0] += float(xy.sum()); d[k][1] += float(gy.sum())
                    name = f"cleave_{T}"
                    if name in trans:
                        trans[name] += np.bincount(kidc.ravel(), xy.ravel(), minlength=K1)
                        expo[name] += np.bincount(kidc.ravel(), gy.ravel(), minlength=K1)
                    h_ = prof.setdefault(f"cleave_entry_to_end_{T}", np.zeros(max_pos))
                    dist = Lm1[:, None] - np.arange(bt.T - 1)[None, :]
                    ok = (dist >= 0) & (dist < max_pos)
                    np.add.at(h_, dist[ok], xy[ok])
        # cleavage entry relative to the read 3' end (entry decided at t, end at L-1)
        for ti, T in enumerate(c.tracks):
            es = lay.edges_of["cleave"]
            es = es[lay.etrack[es] == ti]
            if len(es) and bt.T > 1:
                x = post["xi_d"][:, :, lay.pos_d[es]].sum(2)
                h_ = prof.setdefault(f"cleave_entry_to_end_{T}", np.zeros(max_pos))
                dist = Lm1[:, None] - np.arange(bt.T - 1)[None, :]
                ok = (dist >= 0) & (dist < max_pos)
                np.add.at(h_, dist[ok], x[ok])
        # terminations at read ends, by route
        g_end = post["gamma_h"][rows, Lm1]
        h_end = post["h"][rows, Lm1]
        o = bt.obs[rows, Lm1]
        frac = np.where(o[:, None] == TERM, 1.0,
                        np.where(o[:, None] == MIX,
                                 h_end / np.maximum(h_end + (1 - h_end) * post["csnap"], 1e-300), 0.0))
        tq = g_end * frac * (h_end > 0)
        kid_end = bt.kidx[rows, Lm1] if c.kmer_k else np.zeros(bt.B, int)
        for ti in range(ntr):
            hz = lay.hz_states[lay.track[lay.hz_states] == ti]
            term_random[ti] += np.bincount(kid_end, tq[:, hz].sum(1), minlength=K1)
            if len(lay.cl_states):
                cl = lay.cl_states[lay.track[lay.cl_states] == ti]
                term_cleave[ti] += np.bincount(kid_end, tq[:, cl].sum(1), minlength=K1)
    return dict(trans=trans, expo=expo, prof=prof, term_random=term_random, term_cleave=term_cleave,
                class_track=class_track, cleave_state=cleave_state, n_reads=n_used, W=W, max_pos=max_pos)


CLEAVE_STATES = ("unspliced", "young intron", "in intron", "waiting", "spliced", "later intron")


def cleavage_by_state(tx, support=None, n_kmers=3):
    """Entry into the cleavage path from each splicing state, per track: the learned offset
    (log-odds vs unspliced), the per-nt entry probability with the track's strongest entry
    k-mers (softmax over all transitions of that state, away from splice sites) and, with
    support, the entries inferred in the reads, the exposure (nt spent in that state) and
    entries per 1,000 nt."""
    c, p, lay = tx.cfg, tx.p, tx.lay
    if not c.cleave_substates:
        return {}
    names = kmer_names(c.kmer_k)[:-1]
    K = 4 ** c.kmer_k
    out = {}
    for ti, T in enumerate(c.tracks):
        top = np.argsort(-p["vk_cleave"][ti, :K])[:n_kmers]
        rows = {"state": [], "offset": []}
        for i in top:
            rows[f"P(enter | {names[i]})"] = []
        if support is not None:
            rows.update({"inferred_entries": [], "exposure_nt": [], "entries_per_1000nt": []})
        ref = {"unspliced": T, "young intron": f"{T}y0", "in intron": T + "ps", "waiting": f"{T}w0",
               "spliced": T + "s", "later intron": T + "ps2"}
        for st in CLEAVE_STATES:
            if ref[st] not in lay.idx:
                continue
            s = lay.idx[ref[st]]
            P, outs = _state_probs(tx, lay, s, top)
            j = [k for k, e in enumerate(outs) if lay.kind[e] == "cleave"][0]
            row = {"unspliced": None, "young intron": 0, "in intron": 0, "waiting": 1, "spliced": 2,
                   "later intron": 3}[st]
            rows["state"].append(st)
            rows["offset"].append(0.0 if row is None or "d_cleave" not in p else float(p["d_cleave"][row, ti]))
            for r, i in enumerate(top):
                rows[f"P(enter | {names[i]})"].append(float(P[r, j]))
            if support is not None:
                ent, ex = support.get("cleave_state", {}).get(T, {}).get(st, [0.0, 0.0])
                rows["inferred_entries"].append(ent)
                rows["exposure_nt"].append(ex)
                rows["entries_per_1000nt"].append(1000 * ent / ex if ex > 0 else float("nan"))
        out[T] = {k: (np.array(v) if k != "state" else v) for k, v in rows.items()}
    return out


# ============================================================ report

def transition_report(tx, ds=None, n_reads=2000):
    """Everything above in one dict: baselines, tables per transition group, distributions,
    and (with a dataset) posterior support and profiles."""
    c, p, lay = tx.cfg, tx.p, tx.lay
    sup = transition_support(tx, ds, n_reads) if ds is not None else None
    groups = transition_groups(tx)
    tables = {n: transition_table(tx, n, g, sup) for n, g in groups.items()}
    base = []
    for ti, T in enumerate(c.tracks):
        base.append(dict(kind="random termination", track=T, prob=float(_sig(p["v"][ti, 0])),
                         meaning="per nt in the unspliced base state, no k-mer effect"))
        if c.splicing:
            for kind, b in (("donor", "b_don"), ("acceptor", "b_acc")):
                best = float(np.sort(p["u_" + b[2:]][:, :4], axis=1)[:, -1].sum())
                base.append(dict(kind=kind, track=T, prob=float(_sig(p[b][ti])),
                                 meaning=f"P(recognised) at a site with PWM score 0; "
                                         f"{float(_sig(p[b][ti] + best)):.3f} at the best possible site"))
            pa = float(_sig(p["b_abandon"][ti]))
            base.append(dict(kind="abandon", track=T, prob=pa,
                             meaning=f"per nt with an intron open; mean {1 / pa:,.0f} nt before giving up"))
            if "b_splice" in p:
                ps = float(_sig(p["b_splice"][ti]))
                base.append(dict(kind="splice", track=T, prob=ps,
                                 meaning=f"per nt after the acceptor; mean delay {c.n_wait / ps:,.0f} nt"))
        if "b_switch" in p:
            pw = float(_sig(p["b_switch"][ti]))
            base.append(dict(kind="switch", track=f"{T}->{c.tracks[1 - ti]}", prob=pw,
                             meaning=f"per nt; mean {1 / pw:,.0f} nt between switches"))
    dist = {}
    if c.cleave_substates:
        for ti, T in enumerate(c.tracks):
            pmf, s = cleavage_delay(tx, ti)
            dist[f"cleavage_delay_{T}"] = (pmf, s)
            base.append(dict(kind="cleavage path", track=T, prob=s["within_50nt"],
                             meaning=f"P(terminated within 50 nt of entry); median {s['median']} nt "
                                     f"(10-90%: {s['p10']}-{s['p90']}), advance {s['advance_per_nt']:.3f}/nt, "
                                     f"cut per sub-state {', '.join(f'{x:.2g}' for x in s['cut_per_substate'])}"))
    if c.chain:
        for ci in range(c.promoter_classes):
            pmf, s = early_phase(tx, ci)
            dist[f"early_phase_class{ci}"] = (pmf, s)
            base.append(dict(kind="early phase", track=f"class{ci}", prob=float("nan"),
                             meaning=f"commits {s['mean_commit_nt_after_tss']:.1f} nt after the TSS on "
                                     f"average (median {s['median']}); P(track) at baseline {s['P(track)']}"))
    out = {"baseline": base, "tables": tables, "groups": groups, "distributions": dist, "support": sup,
           "cleavage_by_state": cleavage_by_state(tx, sup)}
    if c.chain:
        out["promoter_chain"] = promoter_chain_table(tx)
    if sup:
        ct = sup["class_track"]
        if c.chain:
            out["posterior_P(track|class)"] = {
                f"class{ci}": dict(zip(c.tracks, (ct[ci] / max(ct[ci].sum(), 1e-300)).round(4).tolist()))
                for ci in range(c.promoter_classes)}
            out["posterior_class_share"] = (ct.sum(1) / max(ct.sum(), 1e-300)).round(4).tolist()
        tr, tc = sup["term_random"].sum(1), sup["term_cleave"].sum(1)
        out["terminations_at_read_ends"] = {T: {"random": float(tr[ti]), "cleavage": float(tc[ti])}
                                            for ti, T in enumerate(c.tracks)}
    return out
