"""TxHMM: one Markov chain along the DNA, nucleotide by nucleotide.

initiation = "chain" (default)

    BG ⟲ --enter_c--> C_c.0 ⟲ -> C_c.1 ⟲ -> ... -> C_c.(m-1) ⟲        promoter class c (sub-states)
     ^                  |  exit (back to BG)    |  initiate (from any C_c.j)
     +------------------+                       v
                                   E_c.0 ⟲ -> ... -> E_c.(e-1) ⟲            early phase of class c:
                                               |  commit (k-mer, per class and track)  no termination,
                                               v                                       no splicing
    track T:  T --donor--> [young intron] --> Tps --acceptor--> Tw --splice--> Ts      (A and P)
              ^               abandon         |
              +-------------------------------+
              any of T, Tps, Tw --cleave--> T_cl0 -> ... -> T_cl(n-1)    cleavage path (unspliced)
              Ts                --cleave--> Ts_cl0 -> ... -> Ts_cl(n-1)  cleavage path (spliced)

  * every transition of the pre-initiation chain, the early phase and the cleavage path is
    driven by a learned k-mer table (+ a bias); no motif or window is built in
  * termination: "random" route = per-track bias + per-track k-mer table + splicing-state
    shift; cleavage route = learned rate per cleavage sub-state
  * splicing keeps its built-in rules: GT/GC and AG gates, PWM windows, minimum intron

initiation = "legacy": INIT -> commit (promoter features) -> tracks; TSS choice by InitModel.

Each read is scored as: P(chain walks `upstream` nt of DNA, initiates at the read's cap,
transcribes the read) / P(initiation within [cap - upstream, cap + init_window_down]),
conditioned on read length >= min_len. With init_score='absolute' there is no initiation
normaliser: P(first firing exactly at the cap) like a first-passage termination. Exact gradients (forward-backward, finite-
difference checked), MAP with L2 priors, minibatch Adam.
"""
import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .sequence import FeatureDef, DEFAULT_INIT_FEATURES

CONT, TERM, CENS, MIX, FREE = 0, 1, 2, 3, 4
_END_TO_OBS = {0: TERM, 1: CENS, 2: MIX, 3: CONT}      # dataprep.END_TYPES -> obs code
L_ALL, L_NOT_S, L_BASE, L_PS, L_WS, L_S, L_PRE, L_E0, L_ANY = range(9)
SUBS = ("base", "ps", "w", "s")
TRANSITION_KINDS = ("commit", "donor", "abandon", "acceptor", "splice", "switch",
                    "cleave", "cleave_adv", "penter", "padv", "pexit", "initiate", "eadv", "ecommit")
TRACK_SUBS = ("base", "ps", "w", "s", "ps2", "w2")
CLEAVE_SUBS = ("cl", "scl")


def _sigmoid(z):
    with np.errstate(over="ignore"):
        return np.exp(-np.logaddexp(0.0, -z))


def _hazard(zh, lay):
    """sigmoid(zh), computed only for the states that can terminate (the others are 0)."""
    h = np.zeros_like(zh)
    hs = np.concatenate([lay.hz_states, lay.cl_states, np.nonzero(np.array(lay.sub) == "init")[0]])
    if len(hs):
        with np.errstate(over="ignore"):
            h[..., hs] = 1.0 / (1.0 + np.exp(-zh[..., hs]))
    return h


# =================================================================== config

@dataclass
class TxConfig:
    tracks: tuple = ("A", "P")
    switching: bool = False
    splicing: bool = True
    n_wait: int = 1
    min_intron: int = 30                # acceptor only >= this many nt after the donor
    donor_dinucs: tuple = ("GT", "GC")
    acceptor_dinucs: tuple = ("AG",)
    donor_window: tuple = (-2, 6)       # relative to t, donor (first intron nt) = t+1
    acceptor_window: tuple = (-19, 3)   # relative to t, acceptor (last intron nt) = t
    # initiation
    initiation: str = "chain"           # 'chain' (learned pre-initiation chain) | 'legacy'
    upstream: int = 100                 # nt of DNA the chain walks before the cap
    start_spread: int = 0               # > 1: Pol II starts in BG uniformly over the first start_spread
                                        # positions of the walk (cap-upstream .. cap-upstream+spread-1)
                                        # instead of exactly at cap-upstream (0/1 = old behaviour)
    init_window_down: int = 100         # P(init at cap) is relative to [cap-upstream, cap+this] (init_score 'window')
    init_score: str = "window"          # 'window': log P(first firing at the cap) - log P(first firing in
                                        # [cap-upstream, cap+init_window_down]) (old behaviour); 'absolute':
                                        # log P(first firing exactly at the cap | Pol II in BG at cap-upstream),
                                        # no normaliser (same first-passage form as termination)
    promoter_classes: int = 2
    promoter_substates: int = 10
    early_substates: int = 3
    upstream_lookahead: int = 3         # pre-initiation transitions at u use the k-mer ending at u+this
    class_track_lean: float = 0.5       # start: class c commit logits to A/P +/- linspace(-lean, lean)
                                        # (0.5: classes lean P / A; 0 = none; < 0 = reversed)
    # termination
    cleave_substates: int = 8           # 0 = no cleavage path
    fix_intron_offsets: object = False  # 'in intron' termination offsets (delta, d_cleave) kept at 0: True / 'all' =
                                        # first (row 0) and later (row 3) introns; 'first' = first intron only
    freeze: tuple = ()                  # parameter blocks kept at their starting values during fit
    all_introns: bool = False           # also track later introns: Ts -donor-> later intron -> Ts
                                        # (own termination offset row 3; splicing parameters shared)
    known_introns: bool = False         # donors / acceptors fire exactly at the reads' known intron
                                        # positions (ReadSet.kptr/kdon/kacc), nowhere else; no PWM,
                                        # no abandon (needs all_introns)
    # sequence inputs (empty = only k-mers; the legacy features are opt-in)
    hazard_features: tuple = ()
    promoter_features: tuple = ()       # legacy initiation only
    dist_tss: bool = False
    dist_scale: float = 1000.0
    kmer_k: int = 5
    kmer_sharing: str = "per_track"     # termination k-mer table: 'per_track' | 'shared'
    kmer_transitions: tuple = ()        # extra k-mer effects on: donor, acceptor, abandon, splice, switch, commit
    hazard_shift: str = "per_track"     # 'per_track' | 'shared' | 'none'
    snapshot: bool = False              # competing harvest hazard (needed for 'mixture' ends)
    supervision: str = "full"           # 'full' | 'status' | 'none'
    max_tokens: int = 150_000           # max reads x positions per batch (memory)
    min_len: int = 1                    # reads shorter than this were filtered out
    l2: dict = field(default_factory=lambda: {
        "v": 1.0, "vk": 10.0, "vkt": 10.0, "u_don": 0.5, "u_acc": 0.5, "w_commit": 1.0, "bias": 1e-3,
        "d_cleave": 1.0})

    def to_json(self):
        d = asdict(self)
        for k in ("hazard_features", "promoter_features"):
            d[k] = [asdict(f) if not isinstance(f, dict) else f for f in getattr(self, k)]
        return json.dumps(d)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        if "initiation" not in d:              # saved before the pre-initiation chain existed
            d["initiation"] = "legacy"
            d["cleave_substates"] = 0
        d.pop("pas_dwell", None); d.pop("pas_window", None)
        for k in ("hazard_features", "promoter_features"):
            d[k] = tuple(FeatureDef(**f) for f in d.get(k, []))
        for k in ("tracks", "donor_dinucs", "acceptor_dinucs", "donor_window", "acceptor_window",
                  "kmer_transitions", "freeze"):
            if k in d:
                d[k] = tuple(d[k])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def chain(self):
        return self.initiation == "chain"


# =================================================================== layout

class Layout:
    """States and transitions of one part of the chain.

    part = 'full'   : everything (prediction, sampling, legacy initiation)
           'prefix' : pre-initiation chain only; E_c.0 are absorbing end states  (chain model)
           'body'   : early phase + tracks, started from E_c.0 at the cap        (chain model)
    Fitting splits every read at its cap: prefix and body are computed separately and
    joined exactly (see TxHMM._bucket), which is much cheaper than carrying all states
    along the whole read.

    compact=True carries the deterministic 'young intron' states as a delay line in the
    forward-backward (identical results, used for fitting). Edges are sorted by source."""

    def __init__(self, cfg: TxConfig, compact=False, part="full"):
        self.cfg, self.part = cfg, part
        if cfg.all_introns and (cfg.switching or not cfg.splicing):
            raise ValueError("all_introns needs splicing and is not implemented with switching")
        if cfg.known_introns and not cfg.all_introns:
            raise ValueError("known_introns needs all_introns")
        chain = cfg.chain
        if part != "full" and not chain:
            raise ValueError("prefix/body layouts only exist for initiation='chain'")
        has_pre = chain and part in ("full", "prefix")
        has_early = chain and part in ("full", "body")
        sinks = chain and part == "prefix"
        has_tracks = part != "prefix"
        n_young = max(cfg.min_intron - 1, 0) if cfg.splicing else 0
        self.compact = bool(compact and has_tracks and n_young > 0 and not cfg.switching)
        self.n_delay = n_young if self.compact else 0
        ntr = len(cfg.tracks)
        C, m, e_n, n_cl = cfg.promoter_classes, cfg.promoter_substates, cfg.early_substates, cfg.cleave_substates
        st = []
        if has_pre:
            st.append(dict(name="BG", track=-1, sub="pre", cls=-1, j=-1))
            for c in range(C):
                for j in range(m):
                    st.append(dict(name=f"C{c}.{j}", track=-1, sub="pre", cls=c, j=j))
        if has_early or sinks:
            for c in range(C):
                for j in range(e_n if has_early else 1):
                    st.append(dict(name=f"E{c}.{j}", track=-1, sub="early", cls=c, j=j))
        if not chain:
            st.append(dict(name="INIT", track=-1, sub="init", cls=-1, j=-1))
        if has_tracks:
            for ti, T in enumerate(cfg.tracks):
                st.append(dict(name=T, track=ti, sub="base", cls=-1, j=-1))
                if cfg.splicing:
                    for k in range(0 if self.compact else n_young):
                        st.append(dict(name=f"{T}y{k}", track=ti, sub="ps", cls=-1, j=-1))
                    st.append(dict(name=T + "ps", track=ti, sub="ps", cls=-1, j=-1))
                    for k in range(cfg.n_wait):
                        st.append(dict(name=f"{T}w{k}", track=ti, sub="w", cls=-1, j=-1))
                    st.append(dict(name=T + "s", track=ti, sub="s", cls=-1, j=-1))
                    if cfg.all_introns:              # later introns (loop back to Ts)
                        for k in range(0 if self.compact else n_young):
                            st.append(dict(name=f"{T}z{k}", track=ti, sub="ps2", cls=-1, j=-1))
                        st.append(dict(name=T + "ps2", track=ti, sub="ps2", cls=-1, j=-1))
                        for k in range(cfg.n_wait):
                            st.append(dict(name=f"{T}v{k}", track=ti, sub="w2", cls=-1, j=-1))
                for k in range(n_cl):
                    st.append(dict(name=f"{T}_cl{k}", track=ti, sub="cl", cls=-1, j=k))
                if cfg.splicing:
                    for k in range(n_cl):
                        st.append(dict(name=f"{T}s_cl{k}", track=ti, sub="scl", cls=-1, j=k))
        self.states = st
        self.S = len(st)
        self.idx = {s["name"]: i for i, s in enumerate(st)}
        I = self.idx
        self.track = np.array([s["track"] for s in st])
        self.sub = [s["sub"] for s in st]
        self.scls = np.array([s["cls"] for s in st])
        self.sj = np.array([s["j"] for s in st])

        E = []                                   # (src, dst, kind, track, cls, j)
        if has_pre:
            E.append((I["BG"], I["BG"], "ref", -1, -1, -1))
            for c in range(C):
                E.append((I["BG"], I[f"C{c}.0"], "penter", -1, c, -1))
                for j in range(m):
                    s_ = I[f"C{c}.{j}"]
                    E.append((s_, s_, "ref", -1, c, j))
                    if j + 1 < m:
                        E.append((s_, I[f"C{c}.{j + 1}"], "padv", -1, c, j))
                    E.append((s_, I["BG"], "pexit", -1, c, j))
                    E.append((s_, I[f"E{c}.0"], "initiate", -1, c, j))
        if has_early:
            for c in range(C):
                for j in range(e_n):
                    s_ = I[f"E{c}.{j}"]
                    E.append((s_, s_, "ref", -1, c, j))
                    if j + 1 < e_n:
                        E.append((s_, I[f"E{c}.{j + 1}"], "eadv", -1, c, j))
                    for ti, T in enumerate(cfg.tracks):
                        E.append((s_, I[T], "ecommit", ti, c, j))
        elif sinks:
            for c in range(C):
                E.append((I[f"E{c}.0"], I[f"E{c}.0"], "ref", -1, c, 0))
        if not chain:
            for ti, T in enumerate(cfg.tracks):
                E.append((I["INIT"], I[T], "commit" if (ntr == 2 and ti == 0) else "ref", ti, -1, -1))
        if has_tracks:
            for ti, T in enumerate(cfg.tracks):
                E.append((I[T], I[T], "ref", ti, -1, -1))
                if cfg.splicing:
                    young = [f"{T}y{k}" for k in range(0 if self.compact else n_young)]
                    chain_ = young + [T + "ps"]
                    E.append((I[T], I[chain_[0]], "donor", ti, -1, -1))   # compact: via the delay line
                    for k in range(len(young)):
                        E.append((I[chain_[k]], I[chain_[k + 1]], "ref", ti, -1, -1))
                    E.append((I[T + "ps"], I[T + "ps"], "ref", ti, -1, -1))
                    E.append((I[T + "ps"], I[T], "abandon", ti, -1, -1))
                    wchain = [f"{T}w{k}" for k in range(cfg.n_wait)] + [T + "s"]
                    E.append((I[T + "ps"], I[wchain[0]], "acceptor", ti, -1, -1))
                    for k in range(cfg.n_wait):
                        E.append((I[f"{T}w{k}"], I[f"{T}w{k}"], "ref", ti, -1, -1))
                        E.append((I[f"{T}w{k}"], I[wchain[k + 1]], "splice", ti, -1, -1))
                    E.append((I[T + "s"], I[T + "s"], "ref", ti, -1, -1))
                    if cfg.all_introns:
                        young2 = [f"{T}z{k}" for k in range(0 if self.compact else n_young)]
                        chain2 = young2 + [T + "ps2"]
                        E.append((I[T + "s"], I[chain2[0]], "donor", ti, -1, -1))
                        for k in range(len(young2)):
                            E.append((I[chain2[k]], I[chain2[k + 1]], "ref", ti, -1, -1))
                        E.append((I[T + "ps2"], I[T + "ps2"], "ref", ti, -1, -1))
                        E.append((I[T + "ps2"], I[T + "s"], "abandon", ti, -1, -1))
                        if cfg.known_introns:            # still waiting at the next known donor
                            for w_ in [f"{T}w{k}" for k in range(cfg.n_wait)] + [f"{T}v{k}" for k in range(cfg.n_wait)]:
                                E.append((I[w_], I[chain2[0]], "donor", ti, -1, -1))
                        wchain2 = [f"{T}v{k}" for k in range(cfg.n_wait)] + [T + "s"]
                        E.append((I[T + "ps2"], I[wchain2[0]], "acceptor", ti, -1, -1))
                        for k in range(cfg.n_wait):
                            E.append((I[f"{T}v{k}"], I[f"{T}v{k}"], "ref", ti, -1, -1))
                            E.append((I[f"{T}v{k}"], I[wchain2[k + 1]], "splice", ti, -1, -1))
                if n_cl:
                    young_names = [f"{T}y{k}" for k in range(0 if self.compact else n_young)] if cfg.splicing else []
                    branches = [(f"{T}_cl", [T] + (young_names + [T + "ps"] + [f"{T}w{k}" for k in range(cfg.n_wait)]
                                                  if cfg.splicing else []))]
                    if cfg.splicing:
                        later = ([f"{T}z{k}" for k in range(0 if self.compact else n_young)] + [T + "ps2"]
                                 + [f"{T}v{k}" for k in range(cfg.n_wait)]) if cfg.all_introns else []
                        branches.append((f"{T}s_cl", [T + "s"] + later))
                    for pre, sources in branches:
                        # (compact layout: the young-intron states enter through the delay line)
                        for src_name in sources:
                            E.append((I[src_name], I[pre + "0"], "cleave", ti, -1, -1))
                        for k in range(n_cl):
                            E.append((I[f"{pre}{k}"], I[f"{pre}{k}"], "ref", ti, -1, k))
                            if k + 1 < n_cl:
                                E.append((I[f"{pre}{k}"], I[f"{pre}{k + 1}"], "cleave_adv", ti, -1, k))
            if cfg.switching and ntr == 2:
                for ti, T in enumerate(cfg.tracks):
                    U = cfg.tracks[1 - ti]
                    for s in st:
                        if s["track"] == ti:
                            E.append((I[s["name"]], I[U + s["name"][len(T):]], "switch", ti, -1, -1))
        E = [E[i] for i in sorted(range(len(E)), key=lambda i: (E[i][0], i))]   # by source
        self.edges = E
        self.E = len(E)
        self.src = np.array([e[0] for e in E])
        self.dst = np.array([e[1] for e in E])
        self.kind = [e[2] for e in E]
        self.etrack = np.array([e[3] for e in E])
        self.ecls = np.array([e[4] for e in E])
        self.ej = np.array([e[5] for e in E])
        n_out = np.bincount(self.src, minlength=self.S)
        assert np.all(n_out > 0), "every state needs an outgoing edge"
        self.src_start = np.searchsorted(self.src, np.arange(self.S))
        self.Msrc = np.eye(self.S)[self.src]
        self.Mdst = np.eye(self.S)[self.dst]
        self.out = [np.nonzero(self.src == s)[0] for s in range(self.S)]
        self.edges_of = {k: np.array([i for i, kk in enumerate(self.kind) if kk == k], int)
                         for k in TRANSITION_KINDS}

        sub = np.array(self.sub)
        pre = sub == "pre"
        post = ~pre
        spliced = np.isin(sub, ["s", "scl", "ps2", "w2"])         # first intron already spliced
        tab = np.zeros((9, self.S), bool)
        tab[L_ALL] = post
        if cfg.splicing:
            tab[L_NOT_S] = post & ~spliced
            tab[L_BASE] = (sub == "base") | (sub == "early")
            tab[L_PS] = sub == "ps"
            tab[L_WS] = (sub == "w") | spliced
            tab[L_S] = spliced
        else:
            tab[L_NOT_S] = tab[L_BASE] = tab[L_PS] = tab[L_WS] = tab[L_S] = post
        tab[L_PRE] = pre
        tab[L_E0] = (sub == "early") & (self.sj == 0)
        tab[L_ANY] = True
        self.label_table = tab

        # which states use which termination parameters
        self.hz_states = np.array([s for s in range(self.S) if self.sub[s] in TRACK_SUBS], int)
        self.cl_states = np.array([s for s in range(self.S) if self.sub[s] in CLEAVE_SUBS], int)
        self.hz_onehot = np.zeros((self.S, ntr))
        self.hz_onehot[self.hz_states, self.track[self.hz_states]] = 1.0
        self.e0_states = np.array([s for s in range(self.S) if self.sub[s] == "early" and self.sj[s] == 0], int)

        self.delayed = np.zeros(self.E, bool)
        if self.compact:
            self.delayed[self.edges_of["donor"]] = True
            # one delay line ("lane") per donor edge: T -> Tps (first intron) and, with
            # all_introns, Ts -> Tps2 (later introns), for every track
            self.don_edge = np.array(self.edges_of["donor"])
            self.base_state = self.src[self.don_edge]
            self.ps_state = self.dst[self.don_edge]
            self.lane_track = self.etrack[self.don_edge]
            self.lane_row = np.array([OFFSET_ROWS[self.sub[s]] for s in self.ps_state])
        self.Msrc_nd = self.Msrc * ~self.delayed[:, None]
        self.Mdst_nd = self.Mdst * ~self.delayed[:, None]

        # 'dynamic' states have an outgoing transition whose probability depends on the
        # position (k-mer, splice site, read); the others (e.g. the cleavage sub-states) have
        # constant transitions, kept as one S x S matrix instead of per position.
        dyn_kinds = {"donor", "acceptor", "commit", "penter", "padv", "pexit", "initiate",
                     "ecommit", "cleave"} | set(cfg.kmer_transitions)
        has_ref = np.zeros(self.S, bool)
        dyn_s = np.zeros(self.S, bool)
        for e_, k in enumerate(self.kind):
            has_ref[self.src[e_]] |= k == "ref"
            dyn_s[self.src[e_]] |= k in dyn_kinds
        assert has_ref.all(), "every state needs a reference (logit 0) edge"
        self.dyn_state = dyn_s
        dmask = dyn_s[self.src]
        self.ed, self.es = np.nonzero(dmask)[0], np.nonzero(~dmask)[0]
        self.pos_d = np.full(self.E, -1); self.pos_d[self.ed] = np.arange(len(self.ed))
        self.pos_s = np.full(self.E, -1); self.pos_s[self.es] = np.arange(len(self.es))
        self.src_d, self.dst_d = self.src[self.ed], self.dst[self.ed]
        self.src_s, self.dst_s = self.src[self.es], self.dst[self.es]
        u = np.unique(self.src_d)
        self.ss_d, self.inv_d = np.searchsorted(self.src_d, u), np.searchsorted(u, self.src_d)
        u = np.unique(self.src_s)
        self.ss_s, self.inv_s = np.searchsorted(self.src_s, u), np.searchsorted(u, self.src_s)
        nd = ~self.delayed[self.ed]
        self.Mdst_d, self.Msrc_d = self.Mdst[self.ed], self.Msrc[self.ed]
        self.Mdst_d_nd, self.Msrc_d_nd = self.Mdst_d * nd[:, None], self.Msrc_d * nd[:, None]
        if self.compact:
            self.de_d = self.pos_d[self.don_edge]
        # compact layout with a cleavage path: young-intron mass can leave the delay line into
        # the (unspliced) cleavage path, as the explicit young states do through their edges
        self.young_cleave = bool(self.compact and cfg.cleave_substates)
        if self.young_cleave:
            self.cl0_state = np.array([I[f"{cfg.tracks[tr]}_cl0"] if self.sub[b] == "base"
                                       else I[f"{cfg.tracks[tr]}s_cl0"]
                                       for b, tr in zip(self.base_state, self.lane_track)])
        self._cs = None                          # compiled edge parameters (TxHMM._compile)


# ================================================================== read sets

@dataclass
class ReadSet:
    """What the model needs about each read (or each hypothetical transcript)."""
    store: object
    gstart: np.ndarray            # global index of the TSS
    L: np.ndarray                 # length (TSS .. 3' end inclusive)
    end_type: np.ndarray          # dataprep.END_TYPES codes
    weight: np.ndarray
    nj: np.ndarray                # number of junctions
    d_rel: np.ndarray             # first junction donor, relative to TSS (-1: none)
    a_rel: np.ndarray             # first junction acceptor, relative to TSS
    canon: np.ndarray             # first junction canonical
    cut: np.ndarray               # read was cut at max_len (splice status at end unknown)
    kptr: np.ndarray = None       # known introns of read i: kdon[kptr[i]:kptr[i+1]] (first intron nt,
    kdon: np.ndarray = None       #   relative to the TSS, >= 1) .. kacc (last intron nt); None = none
    kacc: np.ndarray = None

    def __len__(self):
        return len(self.L)

    @classmethod
    def from_dataset(cls, ds):
        d, a, c = ds.first_junction()
        ptr = ds.reads["jptr"]
        cut = ds.reads.get("cut", np.zeros(len(ds), bool)).astype(bool)
        rs = cls(ds.store, ds.gstart(), ds.lengths.astype(np.int64), ds.reads["end_type"],
                 ds.reads["weight"].astype(float), ptr[1:] - ptr[:-1], d, a, c, cut)
        kin = getattr(ds, "kin", None)
        if kin is not None:                     # the locus's known introns downstream of each TSS
            loc, tss = ds.reads["locus"], ds.reads["tss"]
            k0, k1 = kin["ptr"][loc], kin["ptr"][loc + 1]
            n = k1 - k0
            rid = np.repeat(np.arange(len(loc)), n)
            j = np.repeat(k0 - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(n.sum())
            dr, ar = kin["don"][j] - tss[rid], kin["acc"][j] - tss[rid]
            ok = dr >= 1
            rid, dr, ar = rid[ok], dr[ok], ar[ok]
            rs.kptr = np.concatenate([[0], np.cumsum(np.bincount(rid, minlength=len(loc)))]).astype(np.int64)
            rs.kdon, rs.kacc = dr.astype(np.int64), ar.astype(np.int64)
        return rs

    @classmethod
    def transcripts(cls, store, gstart, L, introns=None):
        """Unlabelled, open-ended stretches (prediction / sampling). introns: optional list of
        (first, last) intron nt relative to gstart, shared by every stretch."""
        n = len(gstart)
        z = np.zeros(n, np.int64)
        rs = cls(store, np.asarray(gstart, np.int64), np.asarray(L, np.int64),
                 np.full(n, 3), np.ones(n), z, z - 1, z - 1, np.zeros(n, bool), np.zeros(n, bool))
        if introns is not None:
            # keep every intron: positions outside the stretch are ignored by make_batch, and an
            # acceptor must survive even when its donor lies before gstart (chunked prediction)
            iv = np.array(list(introns), np.int64).reshape(-1, 2)
            rs.kptr = np.arange(n + 1, dtype=np.int64) * len(iv)
            rs.kdon, rs.kacc = np.tile(iv[:, 0], n), np.tile(iv[:, 1], n)
        return rs

    def subset(self, idx):
        idx = np.asarray(idx)
        out = ReadSet(self.store, *[getattr(self, k)[idx] for k in
                                    ("gstart", "L", "end_type", "weight", "nj", "d_rel",
                                     "a_rel", "canon", "cut")])
        if self.kptr is not None:
            n = self.kptr[idx + 1] - self.kptr[idx]
            j = np.repeat(self.kptr[idx] - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(n.sum())
            out.kptr = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
            out.kdon, out.kacc = self.kdon[j], self.kacc[j]
        return out


class Batch:
    pass


class _Compiled:
    pass


# transitions whose logit is (bias + optional k-mer table), handled generically
BIAS_NAMES = ("b_penter", "b_padv", "b_pexit", "b_init", "b_eadv", "b_ecommit", "b_cleave",
              "b_cleave_adv", "b_abandon", "b_splice", "b_switch", "d_cleave")
# termination offset rows (delta, d_cleave): first intron, waiting (past a 3' ss; shared by first and
# later introns), spliced (exon after the first splice), later intron (only with all_introns)
OFFSET_ROWS = {"ps": 0, "w": 1, "s": 2, "ps2": 3, "w2": 1}
CLEAVE_OFFSET_ROWS = OFFSET_ROWS
KNOWN_LOGIT = 30.0                    # known_introns: donor / acceptor logit at the known positions
TABLE_NAMES = ("vk_penter", "vk_padv", "vk_pexit", "vk_init", "vk_ecommit", "vk_cleave", "vkt")


# ==================================================================== model

class TxHMM:
    def __init__(self, cfg: TxConfig = None, seed=0):
        self.cfg = cfg or TxConfig()
        c = self.cfg
        if c.chain and c.promoter_features:
            raise ValueError("promoter_features are only used with initiation='legacy'")
        if c.chain and not c.kmer_k:
            raise ValueError("the pre-initiation chain needs kmer_k > 0")
        if c.cleave_substates and not c.kmer_k:
            raise ValueError("the cleavage path needs kmer_k > 0")
        parts = ("full", "prefix", "body") if c.chain else ("full",)
        self.lays = {pt: Layout(c, part=pt) for pt in parts}                    # explicit chains
        self.lays_fit = {pt: Layout(c, compact=True, part=pt) for pt in parts}  # delay line
        self.lay = self.lays["full"]
        self.rng = np.random.default_rng(seed)
        ntr = len(c.tracks)
        self.Fh = 1 + len(c.hazard_features) + int(c.dist_tss)
        self.Fp = 1 + len(c.promoter_features)
        self.Kd = c.donor_window[1] - c.donor_window[0] + 1
        self.Ka = c.acceptor_window[1] - c.acceptor_window[0] + 1
        self.don_off = np.arange(c.donor_window[0], c.donor_window[1] + 1)
        self.acc_off = np.arange(c.acceptor_window[0], c.acceptor_window[1] + 1)
        self.nkg = 0 if not c.kmer_k else (1 if c.kmer_sharing == "shared" else ntr)
        self.K1 = 4 ** c.kmer_k + 1 if c.kmer_k else 0
        self.hz_mu = np.zeros(len(c.hazard_features)); self.hz_sd = np.ones(len(c.hazard_features))
        self.pr_mu = np.zeros(len(c.promoter_features)); self.pr_sd = np.ones(len(c.promoter_features))
        self.standardized = False
        self.p = self._init_params()
        self._m, self._u, self._t = {}, {}, 0

    # ---- parameters -------------------------------------------------------

    def _init_params(self):
        c, ntr, r = self.cfg, len(self.cfg.tracks), self.rng
        K1 = self.K1
        p = {}
        p["v"] = 0.01 * r.standard_normal((ntr, self.Fh))
        p["v"][:, 0] = [-3.5, -6.5] if ntr == 2 else [-5.0]
        if c.kmer_k:
            p["vk"] = 0.01 * r.standard_normal((self.nkg, K1))
        if not c.chain:
            p["v_init"] = np.array([-6.0])
            if ntr == 2:
                p["w_commit"] = 0.01 * r.standard_normal(self.Fp)
        else:
            C, m, en = c.promoter_classes, c.promoter_substates, c.early_substates
            sym = np.linspace(-0.5, 0.5, C) if C > 1 else np.zeros(1)      # break class symmetry
            p["b_penter"] = -4.0 + sym
            p["vk_penter"] = 0.05 * r.standard_normal((C, K1))
            p["b_padv"] = np.full(C, -1.0)
            p["vk_padv"] = 0.05 * r.standard_normal((C, K1))
            p["b_pexit"] = np.full(C, -5.0)
            p["vk_pexit"] = 0.05 * r.standard_normal((C, K1))
            p["b_init"] = np.full((C, m), -4.0)
            p["vk_init"] = 0.05 * r.standard_normal((C, K1))
            if en > 1:
                p["b_eadv"] = np.full(C, -1.0)
            bc = np.full((C, ntr), -1.0)
            if ntr == 2 and C > 1:
                lean = np.linspace(-c.class_track_lean, c.class_track_lean, C)
                bc[:, 0] += lean; bc[:, 1] -= lean              # classes start leaning A / P
            p["b_ecommit"] = bc
            p["vk_ecommit"] = 0.05 * r.standard_normal((C, ntr, K1))
        if c.splicing:
            p["u_don"] = 0.01 * r.standard_normal((self.Kd, 5)); p["u_don"][:, 4] = 0
            p["u_acc"] = 0.01 * r.standard_normal((self.Ka, 5)); p["u_acc"][:, 4] = 0
            p["b_don"] = np.full(ntr, -6.0)
            p["b_acc"] = np.full(ntr, -4.0)
            p["b_abandon"] = np.full(ntr, -6.0)
            if c.n_wait:
                p["b_splice"] = np.full(ntr, -3.0)
            if c.hazard_shift != "none":
                p["delta"] = np.zeros((4 if c.all_introns else 3, ntr if c.hazard_shift == "per_track" else 1))
        if c.cleave_substates:
            n = c.cleave_substates
            p["b_cleave"] = np.full(ntr, -7.0)
            p["vk_cleave"] = 0.05 * r.standard_normal((ntr, K1))
            if n > 1:
                p["b_cleave_adv"] = np.full(ntr, np.log(0.3 / 0.7))
            p["cut"] = np.tile(np.linspace(-8.0, -1.5, n) if n > 1 else np.array([-1.5]), (ntr, 1))
            if c.splicing:
                # cleavage-entry offset by splicing state (rows: in intron, waiting, spliced), vs unspliced
                p["d_cleave"] = np.zeros((4 if c.all_introns else 3, ntr))
        if c.switching and ntr == 2:
            p["b_switch"] = np.full(ntr, -8.0)
        if c.snapshot:
            p["kappa"] = np.array([np.log(1 / 2000)])
        if c.kmer_transitions:
            bad = set(c.kmer_transitions) - {"commit", "donor", "acceptor", "abandon", "splice", "switch"}
            if bad:
                raise ValueError(f"kmer_transitions not supported for {bad}")
            if "commit" in c.kmer_transitions and c.chain:
                raise ValueError("kmer_transitions 'commit' is for initiation='legacy' (the chain "
                                 "model commits through its own k-mer tables)")
            p["vkt"] = np.zeros((len(c.kmer_transitions), ntr, K1))
        return p

    def _prior_grads(self):
        l2, g = self.cfg.l2, {}
        for k, P in self.p.items():
            if k == "v":
                gg = -l2["v"] * P; gg[:, 0] = -l2["bias"] * P[:, 0]
            elif k == "w_commit":
                gg = -l2["w_commit"] * P; gg[0] = -l2["bias"] * P[0]
            elif k.startswith("vk"):
                gg = -l2.get(k, l2.get("vk", 10.0)) * P
            elif k in ("u_don", "u_acc"):
                gg = -l2[k] * P
            elif k == "d_cleave":
                gg = -l2.get("d_cleave", 1.0) * P
            else:
                gg = -l2["bias"] * P
            g[k] = gg
        return g

    def log_prior(self):
        return sum(0.5 * float(np.sum(g * P)) for g, P in
                   zip(self._prior_grads().values(), self.p.values()))

    def _edge_spec(self, lay, e):
        """Generic edge e -> (bias name, bias index, k-mer table name or None, table index,
        k-mer source: 'cur' = ending at t, 'la' = ending at t + upstream_lookahead), or None."""
        c = self.cfg
        k, tr, cl, j = lay.kind[e], lay.etrack[e], lay.ecls[e], lay.ej[e]
        if k in ("penter", "padv", "pexit"):
            return ("b_" + k, (cl,), "vk_" + k, (cl,), "la")
        if k == "initiate":
            return ("b_init", (cl, j), "vk_init", (cl,), "la")
        if k == "eadv":
            return ("b_eadv", (cl,), None, None, None)
        if k == "ecommit":
            return ("b_ecommit", (cl, tr), "vk_ecommit", (cl, tr), "cur")
        if k == "cleave":
            return ("b_cleave", (tr,), "vk_cleave", (tr,), "cur")
        if k == "cleave_adv":
            return ("b_cleave_adv", (tr,), None, None, None)
        if k in ("abandon", "splice", "switch"):
            bn = {"abandon": "b_abandon", "splice": "b_splice", "switch": "b_switch"}[k]
            if k in c.kmer_transitions:
                return (bn, (tr,), "vkt", (c.kmer_transitions.index(k), tr), "cur")
            return (bn, (tr,), None, None, None)
        return None

    def _edge_extra(self, lay, e):
        """A second bias added to edge e's logit: the cleavage-entry offset of the source
        state's splicing state (young intron and Tps: in intron; Tw: waiting; Ts: spliced)."""
        if lay.kind[e] == "cleave" and "d_cleave" in self.p:
            row = CLEAVE_OFFSET_ROWS.get(lay.sub[lay.src[e]])
            if row is not None:
                return ("d_cleave", (row, lay.etrack[e]))
        return None

    def _compile(self, lay):
        """Index arrays that evaluate all generic edges of a layout in a few vectorised ops."""
        if lay._cs is not None:
            return lay._cs
        p, K1 = self.p, self.K1
        cs = _Compiled()
        cs.bnames = [n for n in BIAS_NAMES if n in p]
        cs.tnames = [n for n in TABLE_NAMES if n in p]
        boff, o = {}, 0
        for n in cs.bnames:
            boff[n] = o; o += p[n].size
        cs.nb = o
        toff, o = {}, 0
        for n in cs.tnames:
            toff[n] = o; o += p[n].size // K1
        cs.nrows = o
        cs.boff, cs.toff = boff, toff
        es, bidx, te = [], [], {"cur": [], "la": []}
        trow = {"cur": [], "la": []}
        for e in range(lay.E):
            sp = self._edge_spec(lay, e)
            if sp is None:
                continue
            bn, bi, tn, ti, src = sp
            es.append(e)
            bidx.append(boff[bn] + int(np.ravel_multi_index(bi, p[bn].shape)))
            if tn is not None:
                te[src].append(e)
                trow[src].append(toff[tn] + int(np.ravel_multi_index(ti, p[tn].shape[:-1])))
        cs.es, cs.bidx = np.array(es, int), np.array(bidx, int)
        es2, bidx2 = [], []
        for e in range(lay.E):
            ex = self._edge_extra(lay, e)
            if ex is not None:
                es2.append(e)
                bidx2.append(boff[ex[0]] + int(np.ravel_multi_index(ex[1], p[ex[0]].shape)))
        cs.es2, cs.bidx2 = np.array(es2, int), np.array(bidx2, int)
        cs.e, cs.rows, cs.pos = {}, {}, {}
        for src in ("cur", "la"):
            rows, pos = np.unique(np.array(trow[src], int), return_inverse=True)
            cs.e[src], cs.rows[src], cs.pos[src] = np.array(te[src], int), rows, pos.ravel()
        lay._cs = cs
        return cs

    def _flat_tables(self, cs):
        return np.concatenate([self.p[n].reshape(-1, self.K1) for n in cs.tnames]) if cs.tnames else None

    # ---- standardisation (legacy features only) ---------------------------------

    def fit_standardization(self, rs: ReadSet, n=200000):
        c = self.cfg
        r = np.random.default_rng(1)
        if c.hazard_features:
            ridx = r.integers(0, len(rs), n)
            t = (r.random(n) * rs.L[ridx]).astype(np.int64)
            F = rs.store.features(c.hazard_features, rs.gstart[ridx] + t)
            self.hz_mu, self.hz_sd = F.mean(0), F.std(0)
            self.hz_sd[self.hz_sd == 0] = 1.0
        if c.promoter_features:
            F = rs.store.features(c.promoter_features, rs.gstart)
            self.pr_mu, self.pr_sd = F.mean(0), F.std(0)
            self.pr_sd[self.pr_sd == 0] = 1.0
        self.standardized = True

    # ---- batch assembly -----------------------------------------------------

    def _read_codes(self, rs, i, code_row, off, supervision):
        """Splicing labels of read i, written at positions off + t (t relative to the TSS)."""
        L = rs.L[i]
        if supervision == "none" or not self.cfg.splicing:
            return
        spliced = rs.nj[i] > 0
        if not spliced:
            code_row[off + 1:off + L] = L_NOT_S
            return
        known = not rs.cut[i]
        d, a = rs.d_rel[i], rs.a_rel[i]
        nw = self.cfg.n_wait
        if (supervision == "full" and rs.canon[i] and d >= 2 and a + 1 <= L - 1
                and a - d >= max(self.cfg.min_intron - 1, 0)):
            code_row[off + 1:off + d] = L_BASE
            code_row[off + d:off + a + 1] = L_PS
            code_row[off + a + 1:off + L] = L_WS
            if known and L - 1 >= a + 1 + nw:
                code_row[off + L - 1] = L_S
        elif known:
            code_row[off + L - 1] = L_S

    _PART = {"prefix": "prefix", "z": "prefix", "raw": "full",
             "read_full": "full", "sel_full": "full", "pinit_full": "full", "z_full": "full"}

    def make_batch(self, rs: ReadSet, idx, mode="read", supervision=None, labels=True, sign=1.0):
        """mode (chain model)
             prefix : the `upstream` nt before the cap, ending with initiation at the cap
             z      : the same walk, initiating anywhere up to cap + init_window_down
             read   : the read from its cap (body: early phase + tracks)
             sel    : pseudo-read 'survives min_len-1 nt after the cap' (length correction)
             raw    : a plain stretch from gstart, all states, no constraints (prediction)
           mode (legacy initiation): read, sel, raw -- all from the INIT state at the TSS.
           read_full / sel_full / pinit_full / z_full: the chain model's terms computed on
           the full state space in one piece (reference for the tests)."""
        c = self.cfg
        idx = np.asarray(idx)
        B = len(idx)
        W, Wd = c.upstream, c.init_window_down
        part = self._PART.get(mode, "body" if c.chain else "full")
        if (part != "full" or mode.endswith("_full")) and not c.chain:
            raise ValueError(f"mode {mode} needs initiation='chain'")
        Lr = rs.L[idx].astype(np.int64)
        gs = rs.gstart[idx].astype(np.int64)
        nsel = max(c.min_len - 1, 1)
        Lb = {"prefix": np.full(B, W + 1), "z": np.full(B, W + Wd + 1), "read": Lr,
              "sel": np.full(B, nsel), "raw": Lr, "read_full": W + Lr, "sel_full": np.full(B, W + nsel),
              "pinit_full": np.full(B, W + 1), "z_full": np.full(B, W + Wd + 1)}[mode]
        off = W if (part == "prefix" or mode.endswith("_full")) else 0
        lay = self.lays[part]
        g0 = gs - off
        T = int(Lb.max())
        bt = Batch()
        bt.B, bt.T, bt.L, bt.mode, bt.off, bt.part = B, T, Lb, mode, off, part
        bt.init = bt.terminal = None
        bt.inject = None
        if c.chain and c.start_spread > 1 and off == W and W > 0:   # walks starting at cap - upstream
            n = min(c.start_spread, W)
            bt.inject = np.zeros(T); bt.inject[:n] = 1.0 / n
        tt = np.arange(T)
        bt.valid = tt[None] < Lb[:, None]
        bt.tmask = tt[None, :-1] < (Lb - 1)[:, None]
        g = g0[:, None] + np.minimum(tt[None], (Lb - 1)[:, None])
        st = rs.store
        tracks = len(lay.hz_states) > 0
        bt.X = None
        if tracks:
            cols = [np.ones((B, T, 1))]
            if c.hazard_features:
                cols.append((st.features(c.hazard_features, g) - self.hz_mu) / self.hz_sd)
            if c.dist_tss:
                cols.append(((tt[None, :] - off)[..., None] / c.dist_scale) * np.ones((B, 1, 1)))
            bt.X = np.concatenate(cols, axis=2)
        pcols = [np.ones((B, 1))]
        if c.promoter_features:
            pcols.append((st.features(c.promoter_features, gs) - self.pr_mu) / self.pr_sd)
        bt.Xp = np.concatenate(pcols, axis=1)
        bt.kidx = st.kmer_index(g, c.kmer_k) if (c.kmer_k and tracks) else None
        bt.kidx_la = (st.kmer_index(g + c.upstream_lookahead, c.kmer_k)
                      if (c.chain and part in ("prefix", "full")) else None)
        if c.splicing and tracks:
            m = bt.tmask
            gd = np.zeros((B, T), bool); ga = np.zeros((B, T), bool)
            if c.known_introns:                       # transitions only at the known positions
                if rs.kptr is not None:
                    n = rs.kptr[idx + 1] - rs.kptr[idx]
                    b_ = np.repeat(np.arange(B), n)
                    j = np.repeat(rs.kptr[idx] - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(n.sum())
                    td, ta = off + rs.kdon[j] - 1, off + rs.kacc[j]    # donor: t -> t+1 enters the intron
                    for tv, msk in ((td, gd), (ta, ga)):
                        ok = (tv >= off) & (tv < T - 1)
                        msk[b_[ok], tv[ok]] = True
                gd[:, :-1] &= m; ga[:, :-1] &= m
            else:
                gd[:, :-1] = st.dinuc_mask(g[:, :-1], 1, c.donor_dinucs) & m
                ga[:, :-1] = st.dinuc_mask(g[:, :-1], -1, c.acceptor_dinucs) & m
            if off:                                   # nothing to splice before the cap
                gd[:, :off] = False; ga[:, :off] = False
            bt.don_b, bt.don_t = np.nonzero(gd)
            bt.acc_b, bt.acc_t = np.nonzero(ga)
            bt.don_win = st.window_bases(g[bt.don_b, bt.don_t], self.don_off)
            bt.acc_win = st.window_bases(g[bt.acc_b, bt.acc_t], self.acc_off)
        # observations and labels
        obs = np.full((B, T), CONT, np.int8)
        code = np.full((B, T), L_ALL, np.int8)
        sup = supervision or c.supervision
        rows = np.arange(B)
        if mode in ("read", "read_full"):
            obs[rows, Lb - 1] = [_END_TO_OBS[int(e)] for e in rs.end_type[idx]]
            if labels:
                for b, i in enumerate(idx):
                    self._read_codes(rs, i, code[b], off, sup)
        if mode == "raw":
            code[:] = L_ANY
        elif mode in ("prefix", "pinit_full", "z", "z_full"):
            obs[:] = FREE
        if mode in ("prefix", "pinit_full", "read_full", "sel_full"):
            obs[:, :W] = FREE
            code[:, :W] = L_PRE
            code[:, W] = L_E0
        elif mode in ("z", "z_full"):
            code[:] = L_ANY
            code[:, 0] = L_PRE
            code[:, W + Wd] = L_ALL                  # initiated by the end of the window
        bt.obs, bt.code = obs, code
        bt.all_free = bool(np.all(obs == FREE))
        bt.nc_b, bt.nc_t = np.nonzero(obs != CONT)       # the few positions that are not 'continues'
        bt.w = sign * rs.weight[idx].astype(float)
        return bt

    # ---- logits ---------------------------------------------------------------

    def _pwm(self, u, win):
        return u[np.arange(win.shape[1])[None, :], win].sum(1)

    def _logits(self, bt, lay):
        """Hazard logits zh (B,T,S); transition probabilities of the dynamic edges Pd
        (B,T,Ed) and of the static edges Ps (Es,)."""
        c, p = self.cfg, self.p
        B, T, S = bt.B, bt.T, lay.S
        zh = np.full((B, T, S), -np.inf)
        if len(lay.hz_states):
            hs = lay.hz_states
            zt = bt.X @ p["v"].T                                         # (B,T,ntr)
            base = zt[..., lay.track[hs]]
            if c.kmer_k:
                kz = p["vk"].T[bt.kidx]                                  # (B,T,nkg)
                kg = np.zeros(len(hs), int) if self.nkg == 1 else lay.track[hs]
                base = base + kz[..., kg]
            if "delta" in p:
                dv = np.zeros(len(hs))
                for i, s in enumerate(hs):
                    if lay.sub[s] != "base":
                        col = 0 if p["delta"].shape[1] == 1 else lay.track[s]
                        dv[i] = p["delta"][OFFSET_ROWS[lay.sub[s]], col]
                base = base + dv
            zh[..., hs] = base
        if len(lay.cl_states):
            cs_ = lay.cl_states
            zh[..., cs_] = p["cut"][lay.track[cs_], lay.sj[cs_]]
        if not c.chain:
            zh[..., lay.idx["INIT"]] = p["v_init"][0]

        pd = lay.pos_d
        zE = np.zeros((B, T, len(lay.ed)))
        zS = np.zeros(len(lay.es))
        for e in lay.edges_of["commit"]:
            zE[:, :, pd[e]] = (bt.Xp @ p["w_commit"])[:, None]
        if c.splicing and len(lay.edges_of["donor"]):
            ed, ea = lay.edges_of["donor"], lay.edges_of["acceptor"]
            zE[..., pd[ed]] = -np.inf
            zE[..., pd[ea]] = -np.inf
            if c.known_introns:                     # certain at the known positions
                for e in ed:
                    zE[bt.don_b, bt.don_t, pd[e]] = KNOWN_LOGIT
                for e in ea:
                    zE[bt.acc_b, bt.acc_t, pd[e]] = KNOWN_LOGIT
            else:
                sd = self._pwm(p["u_don"], bt.don_win)
                for e in ed:
                    zE[bt.don_b, bt.don_t, pd[e]] = p["b_don"][lay.etrack[e]] + sd
                sa = self._pwm(p["u_acc"], bt.acc_win)
                for e in ea:
                    zE[bt.acc_b, bt.acc_t, pd[e]] = p["b_acc"][lay.etrack[e]] + sa
        cs = self._compile(lay)
        if len(cs.es):
            flat = np.concatenate([p[n].ravel() for n in cs.bnames])
            dyn = pd[cs.es] >= 0
            zE[..., pd[cs.es[dyn]]] = flat[cs.bidx[dyn]]
            zS[lay.pos_s[cs.es[~dyn]]] = flat[cs.bidx[~dyn]]
            if len(cs.es2):
                dyn2 = pd[cs.es2] >= 0
                zE[..., pd[cs.es2[dyn2]]] += flat[cs.bidx2[dyn2]]
                zS[lay.pos_s[cs.es2[~dyn2]]] += flat[cs.bidx2[~dyn2]]
            TAB = self._flat_tables(cs)
            for src, kid in (("cur", bt.kidx), ("la", bt.kidx_la)):
                if len(cs.e[src]):
                    kz = TAB[cs.rows[src]].T[kid]                       # (B,T,rows)
                    zE[..., pd[cs.e[src]]] += kz[..., cs.pos[src]]
        if c.known_introns:                         # a known intron is never abandoned
            for e in lay.edges_of["abandon"]:
                if pd[e] >= 0:
                    zE[..., pd[e]] = -np.inf
                else:
                    zS[lay.pos_s[e]] = -np.inf
        for ki, kind in enumerate(c.kmer_transitions):
            if kind not in ("donor", "acceptor", "commit"):
                continue
            for e in lay.edges_of[kind]:
                tab = p["vkt"][ki, lay.etrack[e]]
                if kind == "donor":
                    zE[bt.don_b, bt.don_t, pd[e]] += tab[bt.kidx[bt.don_b, bt.don_t]]
                elif kind == "acceptor":
                    zE[bt.acc_b, bt.acc_t, pd[e]] += tab[bt.kidx[bt.acc_b, bt.acc_t]]
                else:
                    zE[..., pd[e]] += tab[bt.kidx]
        # softmax over each state's outgoing edges (edges are sorted by source). Every state
        # has a 'ref' edge with logit 0, so without overflow the max need not be subtracted.
        if zE.shape[-1]:
            if zE.max() > 300:
                zE -= np.maximum.reduceat(zE, lay.ss_d, axis=-1)[..., lay.inv_d]
            np.exp(zE, out=zE)
            den = np.add.reduceat(zE, lay.ss_d, axis=-1)
            np.reciprocal(den, out=den)
            zE *= den[..., lay.inv_d]
        Ps = zS
        if len(zS):
            Ps = np.exp(zS - np.maximum.reduceat(zS, lay.ss_s)[lay.inv_s])
            Ps /= np.add.reduceat(Ps, lay.ss_s)[lay.inv_s]
        return zh, zE, Ps

    def _logits_full(self, bt, lay):
        """Hazard logits (B,T,S) and all transition probabilities (B,T,E)."""
        zh, Pd, Ps = self._logits(bt, lay)
        P = np.empty((bt.B, bt.T, lay.E))
        P[..., lay.ed] = Pd
        P[..., lay.es] = Ps
        return zh, P

    @staticmethod
    def _static_matrix(lay, Ps):
        M = np.zeros((lay.S, lay.S))
        np.add.at(M, (lay.src_s, lay.dst_s), Ps)
        return M

    def _emissions(self, bt, h, lay):
        snap = self.cfg.snapshot
        c = _sigmoid(self.p["kappa"][0]) if snap else 0.0
        if bt.all_free:
            e = np.ones_like(h)
        else:
            e = 1.0 - h
            if snap:
                e *= 1 - c
            bb, tt = bt.nc_b, bt.nc_t
            if len(bb):
                hh = h[bb, tt]
                o = bt.obs[bb, tt][:, None]
                cens = (1 - hh) * c if snap else np.ones_like(hh)
                e[bb, tt] = np.where(o == TERM, hh, np.where(o == CENS, cens,
                                     np.where(o == MIX, hh + (1 - hh) * c, 1.0)))
        e *= lay.label_table[bt.code]
        return e, c

    # ---- forward-backward (scaled, probability space) --------------------------

    def forward_backward(self, bt, posteriors=True):
        """Returns ll (B,), dead, a_last (normalised forward at each read's last position) and,
        with posteriors: gamma_h (state posteriors incl. delay line), cE (B,T-1,E) =
        E[transition counts] - E[source occupancy] x P (the logit gradient, unweighted),
        r0 = P(obs | start in s) / P(obs), h, csnap.
        bt.init (B,S): initial distribution (default: state 0); bt.terminal (B,S): backward
        vector at each read's last position (default 1; may be signed, prefix only)."""
        lay = self.lays_fit[bt.part]
        if lay.compact:
            return self._fb_delay(bt, lay, posteriors)
        return self._fb_plain(bt, lay, posteriors)

    def _init_vec(self, bt, S):
        if bt.init is not None:
            return bt.init
        v = np.zeros((bt.B, S))
        inj = getattr(bt, "inject", None)
        v[:, 0] = 1.0 if inj is None else inj[0]
        return v

    @staticmethod
    def _first_start(bt):
        """First position with start mass (0 without a start window)."""
        inj = getattr(bt, "inject", None)
        return 0 if inj is None or bt.init is not None else int(np.flatnonzero(inj)[0])

    @staticmethod
    def _inject(bt, at, e, c, t):
        """Start window: add the start mass of position t (in BG, state 0) to the unnormalised
        forward vector at t, on the scale of a_t = alpha_t / prod_{u<t} c_u."""
        inj = getattr(bt, "inject", None)
        if inj is None or t >= len(inj) or inj[t] == 0.0:
            return
        scale = np.exp(-np.log(c[:, :t]).sum(1))
        at[:, 0] += inj[t] * e[:, t, 0] * scale

    def _static_stats(self, a, bh, V, tm):
        """Per read: sum_t a_t(i) V_t+1(j) (B,S,S) and sum_t a_t(i) bh_t(i) (B,S), the only
        sums the gradient needs for edges with constant probabilities."""
        am = a[:, :-1] * tm
        return np.matmul(am.transpose(0, 2, 1), V), (am * bh[:, :-1]).sum(1)

    def _fb_plain(self, bt, lay, posteriors=True):
        zh, Pd, Ps = self._logits(bt, lay)
        Mst = self._static_matrix(lay, Ps)
        h = _hazard(zh, lay)
        e, csnap = self._emissions(bt, h, lay)
        B, T, S = e.shape
        L, sd, dd_ = bt.L, lay.src_d, lay.dst_d
        a = np.zeros((B, T, S)); c = np.ones((B, T))
        a0 = e[:, 0] * self._init_vec(bt, S)
        c0 = a0.sum(1)
        t_start = self._first_start(bt)
        dead = (c0 <= 0) & (t_start == 0)
        c[:, 0] = np.where(dead | (c0 <= 0), 1.0, c0)
        a[:, 0] = a0 / c[:, 0, None]
        Lmin = int(L.min()) if not dead.any() else 0
        for t in range(1, T):
            act = t < L
            ap = a[:, t - 1]
            at = ((ap[:, sd] * Pd[:, t - 1]) @ lay.Mdst_d + ap @ Mst) * e[:, t]
            self._inject(bt, at, e, c, t)
            ct = at.sum(1)
            if t < Lmin and ct.min() > 0:                # every read active and alive
                c[:, t] = ct
                a[:, t] = at / ct[:, None]
                continue
            bad = act & (ct <= 0) & (t >= t_start)
            dead |= bad
            ok = act & ~bad
            c[:, t] = np.where(ok & (ct > 0), ct, 1.0)
            a[:, t] = np.where(ok[:, None], at / c[:, t, None], 0.0)
        ll = np.where(bt.valid, np.log(c), 0.0).sum(1)
        ll[dead] = -np.inf
        a_last = a[np.arange(B), L - 1]
        if not posteriors:
            return dict(ll=ll, dead=dead, a_last=a_last)
        term = np.ones((B, S)) if bt.terminal is None else bt.terminal
        MstT = Mst.T.copy()
        bh = np.zeros((B, T, S))
        bh[:, T - 1] = term
        for t in range(T - 2, -1, -1):
            v = e[:, t + 1] * bh[:, t + 1] / c[:, t + 1, None]
            nb = (Pd[:, t] * v[:, dd_]) @ lay.Msrc_d + v @ MstT
            bh[:, t] = np.where((t < L - 1)[:, None], nb, term)
        live = (~dead)[:, None]
        gamma = a * bh * (bt.valid & live)[..., None]
        V = e[:, 1:] * bh[:, 1:] / c[:, 1:, None]
        tm = (bt.tmask & live)[..., None]
        cE = a[:, :-1][:, :, sd]
        cE *= Pd[:, :-1]
        cE *= tm
        Vd = V[:, :, dd_]
        extra = {}
        if getattr(bt, "want_xi", False):                # posteriors for interpretation
            extra = dict(xi_d=cE * Vd, gamma=gamma, Pd=Pd)
        Vd -= bh[:, :-1][:, :, sd]
        cE *= Vd
        AV, G = self._static_stats(a, bh, V, tm)
        r0 = e[:, 0] * bh[:, 0] / c[:, 0, None]
        return dict(ll=ll, dead=dead, a_last=a_last, gamma_h=gamma, cE=cE, AV=AV, G=G, Ps=Ps,
                    h=h, csnap=csnap, r0=r0, **extra)

    def _young_cleave_prob(self, bt):
        """(B,T,ntr): per-nt probability that a young-intron polymerase enters the cleavage
        path (young states have only 'continue' and 'cleave': a sigmoid of the cleave logit,
        with the in-intron offset)."""
        p, lay = self.p, self.lays_fit[bt.part]
        tr, row = lay.lane_track, lay.lane_row
        z = (p["b_cleave"][tr] + p["d_cleave"][row, tr])[None, None, :] + p["vk_cleave"].T[bt.kidx][..., tr]
        return _sigmoid(z)

    def _fb_delay(self, bt, lay, posteriors=True):
        """Forward-backward where the min_intron-1 'young intron' states of each track are a
        delay line (ring buffer): mass entering through the donor edge reaches Tps exactly
        n_delay positions later, emitting like Tps on the way, and (with a cleavage path)
        leaving it into the cleavage path with the young states' per-nt probability.
        Identical results to the explicit chain."""
        assert bt.terminal is None
        zh, Pd, Ps = self._logits(bt, lay)
        Mst = self._static_matrix(lay, Ps)
        h = _hazard(zh, lay)
        e, csnap = self._emissions(bt, h, lay)
        B, T, S = e.shape
        L, sd, dd_ = bt.L, lay.src_d, lay.dst_d
        D, base_s, ps_s, de = lay.n_delay, lay.base_state, lay.ps_state, lay.de_d
        ntr = len(base_s)                                  # number of delay lanes
        a = np.zeros((B, T, S)); c = np.ones((B, T))
        a0 = e[:, 0] * self._init_vec(bt, S)
        c0 = a0.sum(1)
        t_start = self._first_start(bt)
        dead = (c0 <= 0) & (t_start == 0)
        c[:, 0] = np.where(dead | (c0 <= 0), 1.0, c0)
        a[:, 0] = a0 / c[:, 0, None]
        Y = np.zeros((B, ntr, D))
        head = 0
        yc = lay.young_cleave
        if yc:
            q = self._young_cleave_prob(bt)
            omq = 1.0 - q
            cl0 = lay.cl0_state
            ysum_st = np.zeros((B, T, ntr)); ycomp_st = np.zeros((B, T, ntr))
        Lmin = int(L.min()) if not dead.any() else 0
        for t in range(1, T):
            act = t < L
            ap = a[:, t - 1]
            pred = (ap[:, sd] * Pd[:, t - 1]) @ lay.Mdst_d_nd + ap @ Mst
            if yc:                                       # young mass leaving into the cleavage path
                ys = Y.sum(2)
                ysum_st[:, t - 1] = ys
                ycomp_st[:, t - 1] = Y[:, :, head]
                Y *= omq[:, t - 1, :, None]
                np.add.at(pred, (slice(None), cl0), ys * q[:, t - 1])
            np.add.at(pred, (slice(None), ps_s), Y[:, :, head])     # lanes may share a target
            Y[:, :, head] = ap[:, base_s] * Pd[:, t - 1, de]
            head = (head + 1) % D
            Y *= e[:, t, ps_s][:, :, None]
            at = pred * e[:, t]
            self._inject(bt, at, e, c, t)
            ct = at.sum(1) + Y.sum((1, 2))
            if t < Lmin and ct.min() > 0:                # every read active and alive
                c[:, t] = ct
                a[:, t] = at / ct[:, None]
                Y /= ct[:, None, None]
                continue
            bad = act & (ct <= 0) & (t >= t_start)
            dead |= bad
            ok = act & ~bad
            c[:, t] = np.where(ok & (ct > 0), ct, 1.0)
            a[:, t] = np.where(ok[:, None], at / c[:, t, None], 0.0)
            Y = np.where(ok[:, None, None], Y / c[:, t, None, None], 0.0)
        ll = np.where(bt.valid, np.log(c), 0.0).sum(1)
        ll[dead] = -np.inf
        a_last = a[np.arange(B), L - 1]
        if not posteriors:
            return dict(ll=ll, dead=dead, a_last=a_last)
        MstT = Mst.T.copy()
        bh = np.ones((B, T, S))
        R = np.ones((B, ntr, D))
        o = 0
        yb0 = np.ones((B, T, ntr))
        for t in range(T - 2, -1, -1):
            act = (t < L - 1)
            v = e[:, t + 1] * bh[:, t + 1] / c[:, t + 1, None]
            r = e[:, t + 1, ps_s] / c[:, t + 1, None]
            nb = (Pd[:, t] * v[:, dd_]) @ lay.Msrc_d_nd + v @ MstT
            nb[:, base_s] += Pd[:, t, de] * r * R[:, :, o]
            R *= r[:, :, None]
            R[:, :, o] = v[:, ps_s]
            if yc:
                R *= omq[:, t, :, None]
                R += (q[:, t] * v[:, cl0])[:, :, None]
            o = (o + 1) % D
            bh[:, t] = np.where(act[:, None], nb, 1.0)
            R = np.where(act[:, None, None], R, 1.0)
            yb0[:, t] = R[:, :, o]
        live = (~dead)[:, None]
        gamma = a * bh * (bt.valid & live)[..., None]
        V = e[:, 1:] * bh[:, 1:] / c[:, 1:, None]
        tm = (bt.tmask & live)[..., None]
        cE = a[:, :-1][:, :, sd]
        cE *= Pd[:, :-1]
        cE *= tm
        Vd = V[:, :, dd_]
        want = getattr(bt, "want_xi", False)
        xi_d = cE * Vd if want else None
        Vd -= bh[:, :-1][:, :, sd]
        cE *= Vd
        a_b = a[:, :-1][:, :, base_s]
        P_d = Pd[:, :-1][:, :, de]
        xd = a_b * P_d * (e[:, 1:][:, :, ps_s] / c[:, 1:, None]) * yb0[:, 1:] * tm
        cE[:, :, de] = xd - a_b * bh[:, :-1][:, :, base_s] * P_d * tm
        AV, G = self._static_stats(a, bh, V, tm)
        extra = {}
        if yc:
            # posterior flows through the young states: entries (xd), leaks into the cleavage
            # path, completions into Tps; occupancy = running sum of the flows
            leak = ysum_st[:, :-1] * q[:, :-1] * V[:, :, cl0] * tm
            comp = ycomp_st[:, :-1] * (1.0 - q[:, :-1]) * V[:, :, ps_s] * tm
            gy = np.concatenate([np.zeros((B, 1, ntr)), np.cumsum(xd - leak - comp, axis=1)], axis=1)
            extra["cE_y"] = leak - gy[:, :-1] * q[:, :-1] * tm
            if want:
                extra.update(xi_young=leak, gamma_young=gy * (bt.valid & live)[..., None])
        else:
            cs = np.concatenate([np.zeros((B, 2, ntr)), np.cumsum(xd, axis=1)], axis=1)
            tt = np.arange(T)
            gy = cs[:, tt + 1] - cs[:, np.maximum(tt + 1 - D, 0)]
        if want:                                         # posteriors for interpretation
            xi_d[:, :, de] = xd
            extra.update(xi_d=xi_d, gamma=gamma.copy(), Pd=Pd)
        gamma_h = gamma
        np.add.at(gamma_h, (slice(None), slice(None), ps_s), gy * (bt.valid & live)[..., None])
        r0 = e[:, 0] * bh[:, 0] / c[:, 0, None]
        return dict(ll=ll, dead=dead, a_last=a_last, gamma_h=gamma_h, cE=cE, AV=AV, G=G, Ps=Ps,
                    h=h, csnap=csnap, r0=r0, **extra)

    # ---- gradients ---------------------------------------------------------------

    def data_grads(self, bt, fb=None, w=None):
        """Gradient of sum_b w_b * ll_b (w defaults to bt.w) w.r.t. every parameter."""
        fb = fb or self.forward_backward(bt)
        c, p = self.cfg, self.p
        lay = self.lays_fit[bt.part]
        w = np.where(fb["dead"], 0.0, bt.w if w is None else w)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        if len(lay.hz_states) or len(lay.cl_states) or not c.chain:
            gh, h, cs_ = fb["gamma_h"], fb["h"], fb["csnap"]
            bb, tt = bt.nc_b, bt.nc_t
            if bt.all_free:
                dz = np.zeros_like(h)
            else:                                   # d log emission / d hazard logit
                dz = -h
                if len(bb):
                    hh, o = h[bb, tt], bt.obs[bb, tt][:, None]
                    den = hh + (1 - hh) * cs_
                    with np.errstate(invalid="ignore", divide="ignore"):
                        dz[bb, tt] = np.where(o == TERM, 1 - hh,
                                     np.where(o == CENS, -hh if c.snapshot else 0.0,
                                     np.where(o == MIX, hh * (1 - hh) * (1 - cs_) / np.maximum(den, 1e-300), 0.0)))
            ch = gh * dz
            ch *= w[:, None, None]                                           # (B,T,S)
            if not c.chain:
                g["v_init"][0] = ch[..., lay.idx["INIT"]].sum()
            hs = lay.hz_states
            if len(hs):
                ctr = ch @ lay.hz_onehot                                     # (B,T,ntr)
                g["v"] = np.einsum("btk,btf->kf", ctr, bt.X)
                if c.kmer_k:
                    kg = np.zeros(len(hs), int) if self.nkg == 1 else lay.track[hs]
                    ck = ch[..., hs] @ np.eye(self.nkg)[kg]                  # (B,T,nkg)
                    for j in range(self.nkg):
                        g["vk"][j] = np.bincount(bt.kidx.ravel(), ck[..., j].ravel(), minlength=self.K1)
                if "delta" in p:
                    cst = ch.sum((0, 1))
                    for s in hs:
                        if lay.sub[s] != "base":
                            col = 0 if p["delta"].shape[1] == 1 else lay.track[s]
                            g["delta"][OFFSET_ROWS[lay.sub[s]], col] += cst[s]
            if len(lay.cl_states):
                ccl = ch[..., lay.cl_states].sum((0, 1))
                np.add.at(g["cut"], (lay.track[lay.cl_states], lay.sj[lay.cl_states]), ccl)
            if c.snapshot and not bt.all_free:
                dk = np.full(h.shape, -cs_)
                if len(bb):
                    hh, o = h[bb, tt], bt.obs[bb, tt][:, None]
                    den = hh + (1 - hh) * cs_
                    with np.errstate(invalid="ignore", divide="ignore"):
                        dk[bb, tt] = np.where(o == TERM, 0.0, np.where(o == CENS, 1 - cs_,
                                     np.where(o == MIX, (1 - hh) * cs_ * (1 - cs_) / np.maximum(den, 1e-300), 0.0)))
                g["kappa"][0] = np.nansum(gh * dk * w[:, None, None])

        cE = fb["cE"]
        if cE.shape[1] == 0:
            return g, fb
        pd = lay.pos_d
        cE = cE * w[:, None, None]                                           # (B,T-1,Ed)
        tot = np.zeros(lay.E)                                                # per edge, summed
        tot[lay.ed] = cE.sum((0, 1))
        if len(lay.es):
            AVw = np.tensordot(w, fb["AV"], 1)
            Gw = w @ fb["G"]
            tot[lay.es] = fb["Ps"] * (AVw[lay.src_s, lay.dst_s] - Gw[lay.src_s])
        for e in lay.edges_of["commit"]:
            g["w_commit"] += cE[:, 0, pd[e]] @ bt.Xp
        if c.splicing and len(lay.edges_of["donor"]) and not c.known_introns:
            for kind, bname, uname, bb, tt, win in (
                    ("donor", "b_don", "u_don", bt.don_b, bt.don_t, bt.don_win),
                    ("acceptor", "b_acc", "u_acc", bt.acc_b, bt.acc_t, bt.acc_win)):
                tsum = np.zeros(len(bb))
                for e in lay.edges_of[kind]:
                    ce = cE[bb, tt, pd[e]]
                    g[bname][lay.etrack[e]] += ce.sum()
                    tsum += ce
                K = win.shape[1]
                flat = (np.arange(K)[None, :] * 5 + win).ravel()
                gu = np.bincount(flat, np.repeat(tsum, K), minlength=K * 5).reshape(K, 5)
                gu[:, 4] = 0.0
                g[uname] = gu
        cs = self._compile(lay)
        if len(cs.es):
            gb = np.bincount(cs.bidx, tot[cs.es], minlength=cs.nb)
            if len(cs.es2):
                gb += np.bincount(cs.bidx2, tot[cs.es2], minlength=cs.nb)
            for n in cs.bnames:
                o_ = cs.boff[n]
                g[n] += gb[o_:o_ + p[n].size].reshape(p[n].shape)
            for src, kid in (("cur", bt.kidx), ("la", bt.kidx_la)):
                es = cs.e[src]
                if not len(es):
                    continue
                rows, pos = cs.rows[src], cs.pos[src]
                nR = len(rows)
                cR = cE[..., pd[es]] @ np.eye(nR)[pos]                       # (B,T-1,rows)
                fi = kid[:, :-1, None] + (np.arange(nR) * self.K1)[None, None, :]
                G = np.bincount(fi.ravel(), cR.ravel(), minlength=nR * self.K1).reshape(nR, self.K1)
                for r_, row in enumerate(rows):
                    for n in cs.tnames:                                       # row -> table
                        o_ = cs.toff[n]
                        if o_ <= row < o_ + p[n].size // self.K1:
                            g[n].reshape(-1, self.K1)[row - o_] += G[r_]
                            break
        if "cE_y" in fb:                                  # young intron -> cleavage path
            cy = fb["cE_y"] * w[:, None, None]
            ty = cy.sum((0, 1))
            np.add.at(g["b_cleave"], lay.lane_track, ty)
            np.add.at(g["d_cleave"], (lay.lane_row, lay.lane_track), ty)
            kid = bt.kidx[:, :-1].ravel()
            for ln in range(cy.shape[2]):
                g["vk_cleave"][lay.lane_track[ln]] += np.bincount(kid, cy[..., ln].ravel(), minlength=self.K1)
        for ki, kind in enumerate(c.kmer_transitions):
            if kind not in ("donor", "acceptor", "commit"):
                continue
            for e in lay.edges_of[kind]:
                tr, j = lay.etrack[e], pd[e]
                if kind == "donor":
                    g["vkt"][ki, tr] += np.bincount(bt.kidx[bt.don_b, bt.don_t], cE[bt.don_b, bt.don_t, j], minlength=self.K1)
                elif kind == "acceptor":
                    g["vkt"][ki, tr] += np.bincount(bt.kidx[bt.acc_b, bt.acc_t], cE[bt.acc_b, bt.acc_t, j], minlength=self.K1)
                else:
                    g["vkt"][ki, tr] += np.bincount(bt.kidx[:, :-1].ravel(), cE[..., j].ravel(), minlength=self.K1)
        return g, fb

    # ---- each read's log-likelihood, and its gradient ---------------------------------

    def _bucket(self, rs: ReadSet, idx, supervision=None, grads=True):
        """Log-likelihood of each read in idx (and, with grads, the gradient of their
        weighted sum).

        legacy : read - [survive min_len]
        chain  : log P(read) = read body - [sel body] + log P(initiate at the cap)
                               - log P(initiate within [cap - upstream, cap + init_window_down])
          (the last term only with init_score 'window'; 'absolute' drops it)
          where the body terms start from the class distribution at initiation. The
          pre-initiation part is one short pass shared by all terms, joined exactly to
          the body passes; its gradient uses a signed backward vector rho over the
          classes (rho_c = r_read - r_sel + 1, r = P(body | class c) / P(body))."""
        idx = np.asarray(idx)
        c = self.cfg
        w = rs.weight[idx].astype(float)
        sel = c.min_len > 2
        if not c.chain:
            terms = [(self.make_batch(rs, idx, "read", supervision), 1.0)]
            if sel:
                terms.append((self.make_batch(rs, idx, "sel"), -1.0))
            fbs = [self.forward_backward(bt, posteriors=grads) for bt, _ in terms]
            dead = np.any([f["dead"] for f in fbs], axis=0)
            ll = sum(s * f["ll"] for (_, s), f in zip(terms, fbs))
            ll[dead] = -np.inf
            if not grads:
                return ll, None
            wl = np.where(dead, 0.0, w)
            g = None
            for (bt, s), f in zip(terms, fbs):
                gi, _ = self.data_grads(bt, f, s * wl)
                g = gi if g is None else {k: g[k] + gi[k] for k in g}
            return ll, g
        B = len(idx)
        pre, body = self.lays_fit["prefix"], self.lays_fit["body"]
        bp = self.make_batch(rs, idx, "prefix")
        fp = self.forward_backward(bp, posteriors=False)
        pi = fp["a_last"][:, pre.e0_states]                         # class at initiation
        parts = []
        for mode, s in (("read", 1.0), ("sel", -1.0)) if sel else (("read", 1.0),):
            bt = self.make_batch(rs, idx, mode, supervision)
            bt.init = np.zeros((B, body.S)); bt.init[:, body.e0_states] = pi
            parts.append((bt, self.forward_backward(bt, posteriors=grads), s))
        window = self._init_window()
        if window:
            bz = self.make_batch(rs, idx, "z")
            fz = self.forward_backward(bz, posteriors=grads)
        dead = fp["dead"] | (fz["dead"] if window else False)
        for _, f, _ in parts:
            dead = dead | f["dead"]
        ll = fp["ll"] - (fz["ll"] if window else 0.0) + sum(s * f["ll"] for _, f, s in parts)
        ll[dead] = -np.inf
        if not grads:
            return ll, None
        wl = np.where(dead, 0.0, w)
        rho = sum(s * f["r0"][:, body.e0_states] for _, f, s in parts) + (1.0 if sel else 0.0)
        bp.terminal = np.zeros((B, pre.S))
        bp.terminal[:, pre.e0_states] = np.where(dead[:, None], 0.0, rho)
        g, _ = self.data_grads(bp, self.forward_backward(bp), wl)
        for bt, f, s in parts:
            gi, _ = self.data_grads(bt, f, s * wl)
            g = {k: g[k] + gi[k] for k in g}
        if window:
            gi, _ = self.data_grads(bz, fz, -wl)
            g = {k: g[k] + gi[k] for k in g}
        return ll, g

    def _init_window(self):
        """True: the initiation term is normalised over [cap - upstream, cap + init_window_down]."""
        v = getattr(self.cfg, "init_score", "window")
        if v not in ("window", "absolute"):
            raise ValueError(f"init_score must be 'window' or 'absolute', not {v!r}")
        return v == "window"

    def _reference_ll(self, rs: ReadSet, idx, supervision=None):
        """Chain model: the same per-read log-likelihood computed on the full state space in
        one piece (slow; for tests)."""
        idx = np.asarray(idx)
        terms = [(self.make_batch(rs, idx, "read_full", supervision), 1.0)]
        if self.cfg.min_len > 2:
            terms += [(self.make_batch(rs, idx, "sel_full"), -1.0),
                      (self.make_batch(rs, idx, "pinit_full"), 1.0)]
        if self._init_window():
            terms.append((self.make_batch(rs, idx, "z_full"), -1.0))
        return sum(s * self.forward_backward(bt, posteriors=False)["ll"] for bt, s in terms)

    def _buckets(self, L, batch_size, rng=None, max_tokens=None):
        max_tokens = max_tokens or self.cfg.max_tokens
        L = np.asarray(L)
        order = np.argsort(L + (rng.random(len(L)) if rng is not None else 0))
        out, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, L[i])
            if cur and (len(cur) >= batch_size or m * (len(cur) + 1) > max_tokens):
                out.append(np.array(cur)); cur, m = [], L[i]
            cur.append(i); cur_max = m
        if cur:
            out.append(np.array(cur))
        return out

    def _bucket_ll(self, rs, idx, supervision=None):
        return self._bucket(rs, idx, supervision, grads=False)[0]

    def per_read_ll(self, rs: ReadSet, supervision=None, chunk=64):
        out = np.zeros(len(rs))
        for idx in self._buckets(rs.L, chunk):
            out[idx] = self._bucket_ll(rs, idx, supervision)
        return out

    def log_posterior(self, rs: ReadSet, supervision=None):
        return float(np.sum(self.per_read_ll(rs, supervision) * rs.weight)) + self.log_prior()

    def ll_per_nt(self, rs, supervision=None):
        return float(np.sum(self.per_read_ll(rs, supervision) * rs.weight) / np.sum(rs.L * rs.weight))

    # ---- optimiser --------------------------------------------------------------------

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, gk in grads.items():
            if k not in self._m:
                self._m[k], self._u[k] = np.zeros_like(gk), np.zeros_like(gk)
            self._m[k] = b1 * self._m[k] + (1 - b1) * gk
            self._u[k] = b2 * self._u[k] + (1 - b2) * gk ** 2
            mh = self._m[k] / (1 - b1 ** self._t)
            uh = self._u[k] / (1 - b2 ** self._t)
            self.p[k] = self.p[k] + lr * mh / (np.sqrt(uh) + eps)
        if "u_don" in self.p:
            self.p["u_don"][:, 4] = 0.0; self.p["u_acc"][:, 4] = 0.0

    def _batch_grads(self, rs, idx):
        """Data gradient of all likelihood terms for one bucket of reads."""
        idx = np.asarray(idx)
        ll, g = self._bucket(rs, idx)
        w = rs.weight[idx]
        return g, float(np.sum(ll * w)), float(np.sum(rs.L[idx] * w)), float(w.sum())

    def _timed_batch_grads(self, rs, idx):
        import time
        t0 = time.perf_counter()
        out = self._batch_grads(rs, idx)
        return out + (time.perf_counter() - t0,)

    def _fixed_offset_rows(self):
        """Rows of delta / d_cleave kept at 0 by fix_intron_offsets (0 = first intron, 3 = later introns)."""
        f = self.cfg.fix_intron_offsets
        if f in (False, None, "", "none"):
            return []
        if f == "first":
            return [0]
        if f in (True, "all"):
            return [0, 3]
        raise ValueError(f"fix_intron_offsets must be False, True, 'all' or 'first', not {f!r}")

    def fit(self, rs: ReadSet, n_iter=600, batch_size=48, lr=0.05, lr_final=0.005,
            val: ReadSet = None, log_every=100, seed=0, verbose=True, n_workers=1,
            tasks_per_worker=2, val_max_reads=5000, patience=0, tol=None, patience_metric="per_nt",
            init_from_data=True):
        """Minibatch Adam. With n_workers > 1, every step draws n_workers x tasks_per_worker
        buckets of reads, processed in parallel, and averages their gradients.
        The validation line uses at most val_max_reads held-out reads. patience_metric:
        'per_nt' (val_ll_per_nt, default tol 1e-6) or 'per_read' (val_ll_per_read, default tol
        1e-3 nats; does not shrink with read length). patience > 0 (needs
        val): stop when that validation log-likelihood has not improved by more than
        tol for `patience` log points, keeping the best parameters."""
        import time
        if patience_metric not in ("per_nt", "per_read"):
            raise ValueError(f"patience_metric must be 'per_nt' or 'per_read', not {patience_metric!r}")
        if tol is None:
            tol = 1e-6 if patience_metric == "per_nt" else 1e-3
        t0 = time.perf_counter()
        if not self.standardized:
            self.fit_standardization(rs)
        if self._t == 0 and init_from_data:
            self.init_from_data(rs, verbose)
        for k in ("delta", "d_cleave"):                 # fixed intron rows start at 0 (also after --init-from)
            if k in self.p:
                self.p[k][[r for r in self._fixed_offset_rows() if r < self.p[k].shape[0]]] = 0.0
        t1 = time.perf_counter()
        rng = np.random.default_rng(seed)
        if val is not None and len(val) > val_max_reads:
            val = val.subset(np.sort(rng.choice(len(val), val_max_reads, replace=False)))
        buckets = self._buckets(rs.L, batch_size, rng)
        sizes = np.array([len(b) for b in buckets], float)
        prob = sizes / sizes.sum()
        w_total = rs.weight.sum()
        t2 = time.perf_counter()
        pool = self._start_pool(n_workers, {"train": rs, "val": val}) if n_workers > 1 else None
        if verbose:
            print(f"[time] fit setup: feature standardisation {t1 - t0:.1f} s, length buckets "
                  f"{t2 - t1:.1f} s, starting {n_workers} worker(s) {time.perf_counter() - t2:.1f} s",
                  flush=True)
        hist = []
        t_train, busy, nt_done = 0.0, 0.0, 0.0
        best_val, best_p, bad = -np.inf, None, 0
        try:
            for it in range(n_iter):
                t_step = time.perf_counter()
                n_tasks = 1 if pool is None else n_workers * tasks_per_worker
                picks = rng.choice(len(buckets), size=n_tasks, p=prob)
                if pool is None:
                    res = [self._timed_batch_grads(rs, buckets[picks[0]])]
                else:
                    picks = sorted(picks, key=lambda b: -rs.L[buckets[b]].max())
                    res = pool.map(_pool_grads, [(self.p, "train", buckets[b]) for b in picks], chunksize=1)
                t_train += time.perf_counter() - t_step
                busy += sum(r[4] for r in res)
                nt_done += sum(r[2] for r in res)
                g = {k: sum((w_total / max(r[3], 1e-12)) * r[0][k] for r in res) / len(res) for k in self.p}
                pg = self._prior_grads()
                grads = {k: g[k] + pg[k] for k in g}
                for k in self.cfg.freeze:               # frozen blocks keep their starting values
                    if k in grads:
                        grads[k][...] = 0.0
                for k in ("delta", "d_cleave"):         # fixed intron rows stay at 0
                    if k in grads:
                        grads[k][[r for r in self._fixed_offset_rows() if r < grads[k].shape[0]]] = 0.0
                lr_t = lr * (lr_final / lr) ** (it / max(n_iter - 1, 1))
                self._adam(grads, lr_t)
                if (it % log_every == 0 or it == n_iter - 1):
                    rec = {"iter": it, "batch_ll_per_nt": sum(r[1] for r in res) / sum(r[2] for r in res),
                           "s_per_step": t_train / (it + 1),
                           "worker_use": busy / (max(n_workers, 1) * t_train),
                           "read_nt_per_s": nt_done / t_train}
                    if val is not None:
                        vr = (self._ll_parallel(pool, "val", val) if pool is not None
                              else self.per_read_ll(val))
                        rec["val_ll_per_nt"] = float(np.sum(vr * val.weight) / np.sum(val.L * val.weight))
                        rec["val_ll_per_read"] = float(np.sum(vr * val.weight) / np.sum(val.weight))
                    hist.append(rec)
                    if verbose:
                        print("  " + "  ".join(f"{k} {v:.5f}" if isinstance(v, float) else f"{k} {v}"
                                               for k, v in rec.items()), flush=True)
                    if patience and val is not None:
                        score = rec["val_ll_" + patience_metric]
                        if score > best_val + tol:
                            best_val, best_p, bad = score, {k: v.copy() for k, v in self.p.items()}, 0
                        else:
                            bad += 1
                            if bad >= patience:
                                if verbose:
                                    print(f"  early stop at iter {it}: no improvement for {patience} log points",
                                          flush=True)
                                break
        finally:
            if pool is not None:
                pool.terminate()
        if patience and best_p is not None:
            self.p = best_p
        return hist

    def init_from_model(self, path, verbose=False):
        """Start from another fitted model's parameters (same blocks and shapes); returns the
        names of the blocks taken over. Feature standardisation is copied too."""
        src = TxHMM.load(path)
        taken = []
        for k, v in src.p.items():
            if k in self.p and self.p[k].shape == v.shape:
                self.p[k] = v.copy(); taken.append(k)
            elif (k in ("delta", "d_cleave") and k in self.p and v.shape[0] == 3
                  and self.p[k].shape == (4,) + v.shape[1:]):   # first-intron-only model -> all_introns
                self.p[k][:3] = v; self.p[k][3] = v[0]; taken.append(k)
        self.hz_mu, self.hz_sd, self.pr_mu, self.pr_sd = src.hz_mu, src.hz_sd, src.pr_mu, src.pr_sd
        self.standardized = src.standardized
        if verbose:
            miss = sorted(set(self.p) - set(taken))
            print(f"[fit] started from {path}: {len(taken)} blocks taken over"
                  + (f"; kept at their own start: {miss}" if miss else ""), flush=True)
        return taken

    def init_from_data(self, rs: ReadSet, verbose=False):
        """Starting values of the random-termination biases from the read lengths: the first
        track about one termination per median read length, the second ten times fewer.
        Without it, the first track can start so short-lived that it is never used and the
        fit stays in a one-track solution (seen on simulated data)."""
        med = float(np.median(rs.L))
        ntr = len(self.cfg.tracks)
        self.p["v"][:, 0] = [np.log(1.0 / med), np.log(0.1 / med)][:ntr] if ntr == 2 else [np.log(0.3 / med)]
        if verbose:
            print(f"[fit] start: random termination per nt {np.round(np.exp(self.p['v'][:, 0]), 5).tolist()} "
                  f"(median read length {med:.0f})", flush=True)

    # ---- process pool ------------------------------------------------------------------

    def _start_pool(self, n_workers, sets):
        import multiprocessing as mp
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        stats = (self.hz_mu, self.hz_sd, self.pr_mu, self.pr_sd)
        sets = {k: v for k, v in sets.items() if v is not None}
        return ctx.Pool(n_workers, initializer=_pool_init, initargs=(self.cfg.to_json(), stats, sets))

    def _ll_parallel(self, pool, name, rs, supervision=None, chunk=64):
        buckets = self._buckets(rs.L, chunk)
        parts = pool.map(_pool_ll, [(self.p, name, b, supervision) for b in buckets], chunksize=1)
        out = np.zeros(len(rs))
        for b, v in zip(buckets, parts):
            out[b] = v
        return out

    def per_read_ll_parallel(self, rs: ReadSet, n_workers, supervision=None, chunk=64):
        pool = self._start_pool(n_workers, {"x": rs})
        try:
            buckets = self._buckets(rs.L, chunk)
            parts = pool.map(_pool_ll, [(self.p, "x", b, supervision) for b in buckets], chunksize=1)
        finally:
            pool.terminate()
        out = np.zeros(len(rs))
        for b, v in zip(buckets, parts):
            out[b] = v
        return out

    # ---- prediction helpers ----------------------------------------------------------------

    def initiation_scan(self, store, g_tss, chunk=512):
        """P(the chain, started `upstream` nt before s in the background state, initiates
        exactly at s, via each promoter class): (n, classes)."""
        assert self.cfg.chain
        g_tss = np.asarray(g_tss, np.int64)
        out = np.zeros((len(g_tss), self.cfg.promoter_classes))
        e0 = self.lays_fit["prefix"].e0_states
        for s in range(0, len(g_tss), chunk):
            gg = g_tss[s:s + chunk]
            rs = ReadSet.transcripts(store, gg, np.ones(len(gg), np.int64))
            fb = self.forward_backward(self.make_batch(rs, np.arange(len(gg)), mode="prefix"), posteriors=False)
            out[s:s + chunk] = np.exp(fb["ll"])[:, None] * fb["a_last"][:, e0]
        return out

    def state_groups(self, lay=None):
        """Named groups of states (boolean masks) used to summarise occupancy."""
        lay = lay or self.lay
        sub = np.array(lay.sub)
        g = {"early": sub == "early",
             "intron": np.isin(sub, ["ps", "w", "ps2", "w2"]),
             "spliced": (sub == "s") | (sub == "scl"),
             "cleavage": (sub == "cl") | (sub == "scl")}
        for ti, T in enumerate(self.cfg.tracks):
            g["track_" + T] = lay.track == ti
        return g

    def flux_profile(self, store, g0, N, start=None, inflow=None, chunk=20000, introns=None):
        """One pass along N nt starting at global position g0.
        start (N,): polymerases placed in the background state of the pre-initiation chain at
          each position (chain model): each walks the DNA, fires once (or never), then
          transcribes.
        inflow (N, classes): polymerases entering the early phase of each class directly
          (legacy: the INIT state).
        Returns per position: polymerases still waiting to fire (waiting), firing there per
        class (init), transcribing polymerases present (reach), terminating (term; term_cleave
        = from a cleavage sub-state), occupancy of the state groups (absolute numbers) and
        event fluxes: donor, acceptor, splice (entering a spliced state), cleave (entering the
        cleavage path), commit to each track."""
        lay = self.lay
        S = lay.S
        sub = np.array(lay.sub)
        chain = self.cfg.chain
        if chain:
            entry, ecls = lay.e0_states, lay.scls[lay.e0_states]
        else:
            entry, ecls = np.array([lay.idx["INIT"]]), np.zeros(1, int)
        C = len(entry)
        start = np.zeros(N) if start is None else np.asarray(start, float)
        inflow = np.zeros((N, C)) if inflow is None else np.asarray(inflow, float).reshape(N, -1)
        if start.any() and not chain:
            raise ValueError("start (a pre-initiation walk) needs initiation='chain'")
        pre = sub == "pre"
        bg = lay.idx["BG"] if chain else -1
        groups = self.state_groups(lay)
        Gm = np.stack([groups[k] for k in groups], 1).astype(float)          # (S, n_groups)
        fedges = {"donor": lay.edges_of["donor"], "acceptor": lay.edges_of["acceptor"],
                  "splice": np.nonzero((sub[lay.dst] == "s") & np.isin(sub[lay.src], ["ps", "w", "ps2", "w2"]))[0],
                  "cleave": lay.edges_of["cleave"]}
        ce = lay.edges_of["ecommit"] if chain else np.nonzero(lay.src == entry[0])[0]
        for ti, T in enumerate(self.cfg.tracks):
            fedges["commit_" + T] = ce[lay.track[lay.dst[ce]] == ti]
        ie = lay.edges_of["initiate"]
        ie_cls = [ie[lay.ecls[ie] == c] for c in range(C)] if chain else []
        reach, term, term_cl, waiting = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        init = inflow.copy()
        occ = np.zeros((N, Gm.shape[1]))
        flux = {k: np.zeros(N) for k in fedges}
        cl = np.zeros(S, bool); cl[lay.cl_states] = True
        f = np.zeros(S)
        for c0 in range(0, N, chunk):
            n = min(chunk, N - c0)
            kin = None if introns is None else [(d - c0, a - c0) for d, a in introns]
            rs = ReadSet.transcripts(store, np.array([g0 + c0]), np.array([n + 1]), introns=kin)
            bt = self.make_batch(rs, [0], mode="raw")
            zh, P = self._logits_full(bt, lay)
            h, P = _hazard(zh, lay)[0], P[0]
            for i in range(n):
                t = c0 + i
                if chain:
                    f[bg] += start[t]
                f[entry] += inflow[t, ecls]
                waiting[t] = f[pre].sum()
                reach[t] = f.sum() - waiting[t]
                occ[t] = f @ Gm
                ts = f * h[i]
                term[t] = ts.sum()
                term_cl[t] = ts[cl].sum()
                if t < N - 1:
                    trans = (f - ts)[lay.src] * P[i]
                    for k, ek in fedges.items():
                        if len(ek):
                            flux[k][t] = trans[ek].sum()
                    for c, ek in enumerate(ie_cls):
                        init[t + 1, c] += trans[ek].sum()           # fires: first nt at t + 1
                    f = trans @ lay.Mdst
        return dict(waiting=waiting, init=init, reach=reach, term=term, term_cleave=term_cl,
                    occ={k: occ[:, j] for j, k in enumerate(groups)}, flux=flux,
                    never_fired=float(f[pre].sum()) if chain else 0.0)

    def profile(self, rs: ReadSet, idx=None):
        """Legacy: per transcript (from its TSS, INIT state), probability of reaching t,
        terminating at t, state occupancy and event fluxes."""
        idx = np.arange(len(rs)) if idx is None else np.asarray(idx)
        bt = self.make_batch(rs, idx, mode="raw")
        zh, P = self._logits_full(bt, self.lay)
        h = _hazard(zh, self.lay)
        lay = self.lay
        B, T, S = h.shape
        occ = np.zeros((B, T, S)); reach = np.zeros((B, T)); term = np.zeros((B, T))
        sub = np.array(lay.sub)
        fedges = {"donor": lay.edges_of["donor"], "acceptor": lay.edges_of["acceptor"],
                  "splice": np.nonzero((sub[lay.dst] == "s") & (sub[lay.src] != "s"))[0],
                  "cleave": lay.edges_of["cleave"]}
        flux = {k: np.zeros((B, T)) for k in fedges}
        cur = np.zeros((B, S)); cur[:, 0] = 1.0
        alive = np.ones(B)
        for t in range(T):
            act = t < bt.L
            occ[:, t] = cur
            reach[:, t] = alive * act
            pt = (cur * h[:, t]).sum(1)
            term[:, t] = alive * pt * act
            post = cur * (1 - h[:, t])
            post = post / np.maximum(post.sum(1), 1e-300)[:, None]
            alive = alive * (1 - pt)
            if t < T - 1:
                trans = post[:, lay.src] * P[:, t]
                for k, ek in fedges.items():
                    if len(ek):
                        flux[k][:, t] = alive * trans[:, ek].sum(1) * (t < bt.L - 1)
                nxt = trans @ lay.Mdst
                cur = nxt / np.maximum(nxt.sum(1), 1e-300)[:, None]
        return dict(occ=occ, reach=reach, term=term, flux=flux, h=h, P=P, batch=bt)

    # ---- sampling ---------------------------------------------------------------------------

    def sample(self, rs: ReadSet, rng=None, start_state=None):
        """Simulate one run of the chain per stretch, starting at gstart in state 0 (the
        background state of the chain model; INIT for legacy) or in start_state (B,) (full
        layout indices, e.g. lay.e0_states[class] to start at a TSS).
        Returns, relative to gstart: initiation position (0 if started at a TSS; -1 if it
        never initiated), end position, whether it terminated, the first spliced junction
        (-1 if not spliced at the end), the track, the promoter class and where the
        cleavage path was entered (-1: never)."""
        rng = rng or np.random.default_rng()
        bt = self.make_batch(rs, np.arange(len(rs)), mode="raw")
        zh, P = self._logits_full(bt, self.lay)
        h = _hazard(zh, self.lay)
        lay = self.lay
        B = bt.B
        z = np.zeros(B, int) if start_state is None else np.asarray(start_state, int).copy()
        alive = np.ones(B, bool)
        end = bt.L - 1
        terminated = np.zeros(B, bool)
        started = (start_state is not None) or not self.cfg.chain
        init = np.zeros(B, int) if started else np.full(B, -1)
        don = np.full(B, -1); acc = np.full(B, -1); cl = np.full(B, -1)
        track = np.full(B, -1)
        pcls = lay.scls[z].copy() if start_state is not None else np.full(B, -1)
        kinds = np.array(lay.kind)
        for t in range(bt.T):
            act = alive & (t < bt.L)
            if not act.any():
                break
            ht = h[np.arange(B), t, z]
            die = act & (rng.random(B) < ht)
            end[die] = t; terminated[die] = True
            alive &= ~die
            act = alive & (t < bt.L - 1)
            if t == bt.T - 1 or not act.any():
                continue
            pe = P[:, t] * (lay.src[None, :] == z[:, None])
            cum = np.cumsum(pe, axis=1)
            u = rng.random(B)[:, None] * cum[:, -1:]
            e = np.minimum((cum < u).sum(1), lay.E - 1)
            k = kinds[e]
            don[act & (k == "donor")] = t + 1
            don[act & (k == "abandon")] = -1
            acc[act & (k == "acceptor")] = t
            cl[act & (k == "cleave")] = t
            hit = act & (k == "initiate")
            init[hit] = t + 1
            pcls[hit] = lay.ecls[e[hit]]
            com = act & ((k == "ecommit") | (k == "commit") | ((k == "ref") & (lay.src[e] == 0) & (lay.dst[e] != 0)))
            track[com] = lay.track[lay.dst[e[com]]]
            z[act] = lay.dst[e[act]]
        spliced = np.array([lay.sub[s] in ("s", "scl", "ps2", "w2") for s in z])
        don[~spliced] = -1; acc[~spliced] = -1
        return dict(init=init, end=end, terminated=terminated, donor=don, acceptor=acc,
                    track=track, promoter_class=pcls, cleave=cl)

    # ---- persistence -------------------------------------------------------------------------

    def save(self, path):
        np.savez(path, cfg=np.array(self.cfg.to_json()),
                 hz_mu=self.hz_mu, hz_sd=self.hz_sd, pr_mu=self.pr_mu, pr_sd=self.pr_sd,
                 **{f"p_{k}": v for k, v in self.p.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(TxConfig.from_json(str(z["cfg"])))
        m.hz_mu, m.hz_sd, m.pr_mu, m.pr_sd = z["hz_mu"], z["hz_sd"], z["pr_mu"], z["pr_sd"]
        m.standardized = True
        # parameters added since the model was saved keep their neutral start (e.g. d_cleave = 0)
        m.p.update({k[2:]: z[k] for k in z.files if k.startswith("p_")})
        return m

    def summary(self):
        c, p = self.cfg, self.p
        names = ["bias"] + [f.name for f in c.hazard_features] + (["dist_tss"] if c.dist_tss else [])
        out = {"random_termination": {T: dict(zip(names, np.round(p["v"][i], 3).tolist()))
                                      for i, T in enumerate(c.tracks)}}
        if "delta" in p:
            out["termination_shift"] = {s: np.round(p["delta"][i], 3).tolist()
                                        for i, s in enumerate(("in_intron", "waiting", "spliced", "later_intron"))
                                        if i < len(p["delta"])}
        if "w_commit" in p:
            pn = ["bias"] + [f.name for f in c.promoter_features]
            out["commit_to_" + c.tracks[0]] = dict(zip(pn, np.round(p["w_commit"], 3).tolist()))
        for k in ("b_don", "b_acc", "b_abandon", "b_splice", "b_switch", "b_cleave"):
            if k in p:
                out[k] = dict(zip(c.tracks, np.round(p[k], 3).tolist()))
        if "d_cleave" in p:
            out["cleave_entry_offset (log-odds vs unspliced)"] = {
                T: dict(zip(("in_intron", "waiting", "spliced", "later_intron"), np.round(p["d_cleave"][:, i], 3).tolist()))
                for i, T in enumerate(c.tracks)}
        if "b_cleave_adv" in p:
            out["cleave_advance_per_nt"] = dict(zip(c.tracks, np.round(_sigmoid(p["b_cleave_adv"]), 4).tolist()))
        if "cut" in p:
            out["cleave_cut_per_nt"] = {T: np.round(_sigmoid(p["cut"][i]), 4).tolist() for i, T in enumerate(c.tracks)}
        if "b_ecommit" in p:
            pa = np.exp(p["b_ecommit"]) / np.exp(p["b_ecommit"]).sum(1, keepdims=True)
            out["P(track | promoter class)"] = {f"class{ci}": dict(zip(c.tracks, np.round(pa[ci], 3).tolist()))
                                                for ci in range(c.promoter_classes)}
        if "kappa" in p:
            out["snapshot_hazard"] = float(_sigmoid(p["kappa"][0]))
        return out


# ============================================================ worker processes

_WORKER = {}


def _pool_init(cfg_json, stats, sets):
    m = TxHMM(TxConfig.from_json(cfg_json))
    m.hz_mu, m.hz_sd, m.pr_mu, m.pr_sd = stats
    m.standardized = True
    _WORKER["m"], _WORKER["sets"] = m, sets


def _pool_grads(args):
    p, name, idx = args
    m = _WORKER["m"]
    m.p = p
    return m._timed_batch_grads(_WORKER["sets"][name], idx)


def _pool_ll(args):
    p, name, idx, supervision = args
    m = _WORKER["m"]
    m.p = p
    return m._bucket_ll(_WORKER["sets"][name], idx, supervision)


# ============================================================ gradient check

def check_gradients(model: TxHMM, rs: ReadSet, idx, n_checks=30, eps=1e-6, seed=0):
    """Worst relative error, per parameter block, between the analytic gradient of the log
    posterior (reads idx) and central finite differences."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    w = rs.weight[idx]
    _, g = model._bucket(rs, idx)
    pg = model._prior_grads()

    def obj():
        return float(np.sum(model._bucket(rs, idx, grads=False)[0] * w)) + model.log_prior()

    worst = {}
    for k in model.p:
        full = g[k] + pg[k]
        P = model.p[k]
        cand = np.argwhere(np.abs(full) > 1e-12)
        if len(cand) == 0:
            cand = np.argwhere(np.ones_like(P, bool))
        sel = cand[rng.choice(len(cand), min(n_checks, len(cand)), replace=False)]
        err = 0.0
        for ix in map(tuple, sel):
            old = P[ix]
            P[ix] = old + eps; a = obj()
            P[ix] = old - eps; b = obj()
            P[ix] = old
            num = (a - b) / (2 * eps)
            err = max(err, abs(num - full[ix]) / max(1.0, abs(num)))
        worst[k] = err
    return worst


# ============================================================ initiation model

class InitModel:
    """Relative initiation propensity along a sequence:
         P(TSS = s | a TSS in this window) = softmax_s( PWM(s-40..s+10) + w . feats(s) )
       Trained on cap positions; windows span each locus' caps +/- pad."""

    def __init__(self, window=(-40, 10), features=DEFAULT_INIT_FEATURES, l2=1.0, pad=150,
                 max_loci=2000):
        """max_loci: fit on at most this many randomly chosen loci (the model has ~260
        parameters; more loci only cost time). None = all loci."""
        self.window, self.features, self.l2, self.pad = tuple(window), tuple(features), l2, pad
        self.max_loci = max_loci
        self.off = np.arange(window[0], window[1] + 1)
        self.K, self.F = len(self.off), len(self.features)
        self.u = np.zeros((self.K, 5))
        self.w = np.zeros(self.F)
        self.mu = np.zeros(self.F); self.sd = np.ones(self.F)

    def _candidates(self, ds):
        """Candidate TSS positions: the union of [cap - pad, cap + pad] around every cap,
        grouped by locus (one softmax per locus). Vectorised; caps far apart in a locus do
        not drag in the whole stretch between them."""
        loc, pos = ds.caps["locus"], ds.caps["pos"]
        o = np.lexsort((pos, loc))
        loc, pos = loc[o], pos[o]
        L = ds.store.lengths[loc]
        lo = np.maximum(pos - self.pad, 0)
        hi = np.minimum(pos + self.pad, L - 1)
        # caps are sorted by position within a locus, so `hi` never decreases inside a locus:
        # a new block starts at a new locus or when the next window does not touch the previous
        new = np.ones(len(pos), bool)
        new[1:] = (loc[1:] != loc[:-1]) | (lo[1:] > hi[:-1] + 1)
        blk = np.cumsum(new) - 1
        last = np.r_[new[1:], True]
        b_lo, b_hi, b_loc = lo[new], hi[last], loc[new]
        size = b_hi - b_lo + 1
        b_off = np.concatenate([[0], np.cumsum(size)])
        M = int(b_off[-1])
        within = np.arange(M) - np.repeat(b_off[:-1], size)
        local = np.repeat(b_lo, size) + within
        locus_of = np.repeat(b_loc, size)
        g = ds.store.starts[locus_of] + local
        _, seg = np.unique(locus_of, return_inverse=True)
        cnt = np.bincount(b_off[blk] + pos - b_lo[blk], minlength=M)
        return g, seg.astype(np.int64), cnt.astype(float)

    def _pwm_scores(self, win_t, u):
        """win_t: (K, M) base codes -> sum_k u[k, win_t[k]] without an (M, K) temporary."""
        s = np.zeros(win_t.shape[1])
        for k in range(self.K):
            s += u[k][win_t[k]]
        return s

    def score(self, store, g):
        g = np.asarray(g)
        win_t = np.ascontiguousarray(store.window_bases(g, self.off).T)
        s = self._pwm_scores(win_t, self.u)
        if self.F:
            s = s + ((store.features(self.features, g) - self.mu) / self.sd) @ self.w
        return s

    @staticmethod
    def _seg_softmax(eta, seg, nseg):
        """Softmax within each segment; segments are contiguous runs of `seg`."""
        starts = np.r_[0, np.nonzero(seg[1:] != seg[:-1])[0] + 1]
        m = np.maximum.reduceat(eta, starts)
        ex = np.exp(eta - m[seg])
        z = np.add.reduceat(ex, starts)
        return ex / z[seg], m + np.log(z)

    def fit(self, ds, verbose=True, seed=0):
        from scipy.optimize import minimize
        loci = np.unique(ds.caps["locus"])
        if self.max_loci and len(loci) > self.max_loci:
            keep = np.random.default_rng(seed).choice(loci, self.max_loci, replace=False)
            ds = ds.subset_loci(np.sort(keep))
        g, seg, cnt = self._candidates(ds)
        nseg = seg.max() + 1
        N = np.bincount(seg, cnt, minlength=nseg)
        win_t = np.ascontiguousarray(ds.store.window_bases(g, self.off).T)      # (K, M) uint8
        F = ds.store.features(self.features, g) if self.F else np.zeros((len(g), 0))
        if self.F:
            self.mu, self.sd = F.mean(0), F.std(0); self.sd[self.sd == 0] = 1
            F = (F - self.mu) / self.sd

        def unpack(th):
            u = th[:self.K * 5].reshape(self.K, 5).copy(); u[:, 4] = 0
            return u, th[self.K * 5:]

        def f(th):
            u, w = unpack(th)
            eta = self._pwm_scores(win_t, u) + F @ w
            pr, lse = self._seg_softmax(eta, seg, nseg)
            ll = np.sum(cnt * eta) - np.sum(N * lse)
            r = cnt - N[seg] * pr
            gu = np.stack([np.bincount(win_t[k], r, minlength=5) for k in range(self.K)])
            gu[:, 4] = 0
            gw = F.T @ r
            obj = -ll + 0.5 * self.l2 * np.sum(th ** 2)
            grad = -np.concatenate([gu.ravel(), gw]) + self.l2 * th
            return obj, grad

        th0 = np.zeros(self.K * 5 + self.F)
        res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.u, self.w = unpack(res.x)
        if verbose:
            print(f"[init] fitted on {int(N.sum())} caps in {nseg} loci ({len(g)} candidate positions); "
                  f"LL/cap {-res.fun / N.sum():.3f} (uniform {-np.sum(N * np.log(np.bincount(seg))) / N.sum():.3f})")
        return self

    def evaluate(self, ds):
        g, seg, cnt = self._candidates(ds)
        nseg = seg.max() + 1
        N = np.bincount(seg, cnt, minlength=nseg)
        eta = self.score(ds.store, g)
        pr, lse = self._seg_softmax(eta, seg, nseg)
        ll = (np.sum(cnt * eta) - np.sum(N * lse)) / N.sum()
        unif = -np.sum(N * np.log(np.bincount(seg))) / N.sum()
        # is the predicted mode within 2 nt of the observed mode? (first maximum in each locus)
        o1 = np.lexsort((np.arange(len(g)), -eta, seg))
        o2 = np.lexsort((np.arange(len(g)), -cnt, seg))
        first = np.r_[True, seg[o1][1:] != seg[o1][:-1]]
        hit = np.abs(g[o1][first] - g[o2][first]) <= 2
        return {"ll_per_cap": float(ll), "uniform_ll_per_cap": float(unif),
                "mode_within_2nt": float(np.mean(hit))}

    def save(self, path):
        np.savez(path, u=self.u, w=self.w, mu=self.mu, sd=self.sd, window=np.array(self.window),
                 l2=self.l2, pad=self.pad, max_loci=self.max_loci or 0,
                 features=np.array(json.dumps([asdict(f) for f in self.features])))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        feats = tuple(FeatureDef(**f) for f in json.loads(str(z["features"])))
        m = cls(tuple(z["window"]), feats, float(z["l2"]), int(z["pad"]),
                int(z["max_loci"]) if "max_loci" in z.files else 2000)
        m.u, m.w, m.mu, m.sd = z["u"], z["w"], z["mu"], z["sd"]
        return m
