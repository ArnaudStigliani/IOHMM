"""Synthetic genome + TLDR-seq-like reads with a known ground truth.

Default truth (truth_models): the chain model, driven by k-mer tables only (no motif,
window or hand-made feature anywhere):
  * pre-initiation: class 0 is entered on TATAA-like k-mers and initiates from its late
    sub-states on Inr-like k-mers (YYANW around the TSS); class 1 is entered on GC-rich
    k-mers and initiates broadly where the TSS nt is a purine after a pyrimidine
  * early phase: class 0 commits mostly to A, class 1 mostly to P
  * random termination: higher on track A, raised by T-rich k-mers
  * cleavage path: entered on AATAA / ATTAA (both tracks), cut ~20-25 nt later
  * splicing: consensus PWMs (the built-in splicing rules)

Loci (own chromosome, random strand): half TATA promoters (TATA box ~30 nt upstream of
an Inr), half GC-rich promoters; 1-3 introns with consensus splice sites; planted
AATAAA and T-runs; background sequence full of decoy GT/AG.

Caps are drawn from the model's initiation propensity (initiation_scan) around the
promoter; each transcript is then simulated nucleotide by nucleotide from the early
phase of its class. Outputs genome.fa, reads.bed12, reads.bam and truth.json.
"""
import json
import os

import numpy as np

from .genome import write_fasta, write_bed12, write_bam, revcomp
from .sequence import SequenceStore, FeatureDef, kmer_names
from .model import TxConfig, TxHMM, InitModel, ReadSet

PREP = dict(end_mode=r"name:\|end=(\w+)", end_types={"term": "terminated", "surv": "survived"})


def _pwm(K, spec):
    u = np.zeros((K, 5))
    for k, d in spec.items():
        for b, val in d.items():
            u[k, "ACGT".index(b)] = val
    return u


def _kmer_ids(k, pred):
    """Indices of the k-mers (strings) for which pred(kmer) is true."""
    names = kmer_names(k)[:-1]
    return np.array([i for i, s in enumerate(names) if pred(s)], int)


def _set_splicing(p):
    # donor window offsets -2..+6 relative to t  <=> positions -3..+6 around the exon|intron boundary
    p["u_don"] = _pwm(9, {0: {"A": .7, "C": .7}, 1: {"A": 1.2}, 2: {"G": 1.5},
                          3: {"G": 0.0}, 4: {"T": 0.0, "C": -2.5},
                          5: {"A": 1.0, "G": 1.0}, 6: {"A": 1.5}, 7: {"G": 2.0}, 8: {"T": 1.0}})
    p["b_don"] = np.array([-9.5, -8.5])        # A-track transcripts splice less
    spec = {k: {"T": 0.7, "C": 0.5} for k in range(0, 15)}   # polypyrimidine tract
    spec[17] = {"C": 1.0, "T": 0.5}                          # Y of YAG
    spec[20] = {"G": 1.0}                                    # first exon nt
    p["u_acc"] = _pwm(23, spec)
    p["b_acc"] = np.array([-10.0, -10.0])
    p["b_abandon"] = np.array([-7.0, -7.0])
    p["b_splice"] = np.array([-4.0, -4.0])     # mean splicing delay ~55 nt


def truth_models(promoter_substates=10, early_substates=3, cleave_substates=8, upstream=100):
    """The k-mer-only truth (chain model). Returns (TxHMM, None)."""
    k = 5
    cfg = TxConfig(tracks=("A", "P"), n_wait=1, kmer_k=k, upstream=upstream, init_window_down=100,
                   promoter_classes=2, promoter_substates=promoter_substates,
                   early_substates=early_substates, cleave_substates=cleave_substates,
                   kmer_sharing="per_track", hazard_shift="shared", snapshot=False)
    tx = TxHMM(cfg)
    tx.standardized = True
    p = tx.p
    m = promoter_substates
    for name in [n for n in p if n.startswith("vk")]:
        p[name][:] = 0.0
    gc = lambda s: sum(ch in "GC" for ch in s)
    # --- class 0: TATA-entered, Inr-initiated
    p["b_penter"] = np.array([-11.0, -9.0])
    p["vk_penter"][0, _kmer_ids(k, lambda s: s == "TATAA")] = 10.0
    p["vk_penter"][0, _kmer_ids(k, lambda s: s in ("TATAT", "TATTA", "ATAAA"))] = 6.0
    p["b_padv"] = np.array([np.log(0.36 / 0.64), -1.0])        # ~25 nt to cross class 0
    p["b_pexit"] = np.array([-7.0, -4.0])
    b_init = np.full((2, m), -6.0)
    late = max(m - 4, 0)
    b_init[0, :late] = -14.0
    b_init[0, late:] = -6.0
    p["b_init"] = b_init
    inr = _kmer_ids(k, lambda s: s[0] in "CT" and s[1] in "CT" and s[2] == "A" and s[4] in "AT")
    p["vk_init"][0, inr] = 4.0
    p["vk_init"][0, _kmer_ids(k, lambda s: s == "TCAGT")] = 6.0
    # --- class 1: GC-entered, YR-initiated
    p["vk_penter"][1, :4 ** k] = 1.2 * (np.array([gc(s) for s in kmer_names(k)[:-1]]) - 3)
    p["vk_pexit"][1, :4 ** k] = -0.6 * (np.array([gc(s) for s in kmer_names(k)[:-1]]) - 3)
    p["vk_init"][1, _kmer_ids(k, lambda s: s[1] in "CT" and s[2] in "AG")] = 2.0
    # --- early phase: class 0 -> A (0.82), class 1 -> P (0.88)
    if "b_eadv" in p:
        p["b_eadv"] = np.array([0.0, 0.0])
    p["b_ecommit"] = np.array([[0.5, -1.0], [-1.5, 0.5]])
    # --- random termination
    p["v"] = np.array([[-5.5], [-9.5]])
    p["vk"][0, _kmer_ids(k, lambda s: s == "TTTTT")] = 3.0
    p["vk"][0, _kmer_ids(k, lambda s: s in ("ATTTT", "TTTTA"))] = 1.5
    p["vk"][1, _kmer_ids(k, lambda s: s == "TTTTT")] = 1.5
    # --- cleavage path
    p["b_cleave"] = np.array([-14.0, -14.0])
    for ti in range(2):
        p["vk_cleave"][ti, _kmer_ids(k, lambda s: s == "AATAA")] = 13.6
        p["vk_cleave"][ti, _kmer_ids(k, lambda s: s == "ATTAA")] = 12.0
    if cleave_substates > 1:
        p["b_cleave_adv"] = np.full(2, np.log(0.35 / 0.65))
    cut = np.full(cleave_substates, -12.0); cut[-1] = -0.7
    p["cut"] = np.tile(cut, (2, 1))
    # --- splicing and the termination shift while an intron is open / after splicing
    _set_splicing(p)
    p["delta"] = np.array([[-1.0], [-1.0], [-1.5]])
    return tx, None


TRUTH_HAZARD_FEATURES = (FeatureDef("pas[-35,-10]", "motif:A[AT]TAAA", -35, -10, "any"),
                         FeatureDef("gc[-40,-1]", "GC", -40, -1, "mean"))
TRUTH_PROMOTER_FEATURES = (FeatureDef("prom_gc[-100,-1]", "GC", -100, -1, "mean"),)


def truth_models_legacy():
    """The first version's truth (legacy initiation + hand-made features); for
    benchmarks/run_simulation.py."""
    cfg = TxConfig(tracks=("A", "P"), n_wait=1, hazard_features=TRUTH_HAZARD_FEATURES,
                   promoter_features=TRUTH_PROMOTER_FEATURES, dist_tss=False, kmer_k=5,
                   kmer_sharing="shared", hazard_shift="shared", snapshot=False,
                   initiation="legacy", cleave_substates=0)
    tx = TxHMM(cfg)
    tx.standardized = True                     # truth works on raw feature values
    p = tx.p
    p["v"] = np.array([[-4.0, 2.5, 0.0],       # A: attenuation-prone, PAS-sensitive
                       [-7.5, 1.0, 0.0]])      # P: productive
    p["v_init"] = np.array([-8.0])
    p["vk"][:] = 0.0
    p["vk"][0, int("".join(str("ACGT".index(c)) for c in "TTTTT"), 4)] = 1.5
    p["w_commit"] = np.array([-5.0, 8.0])      # logit P(A) = -5 + 8 * promoter GC
    _set_splicing(p)
    p["delta"] = np.array([[-1.0], [-1.0], [-1.5]])
    init = InitModel(window=(-40, 10), features=())
    u = np.zeros((51, 5))
    for k, b in enumerate("TATAAA"):          # TATA box at -31..-26
        u[40 - 31 + k, "ACGT".index(b)] = 1.5
    for off, spec in {-2: "CT", -1: "CT", 0: "A", 2: "AT", 3: "CT", 4: "CT"}.items():
        for b in spec:
            u[40 + off, "ACGT".index(b)] = 1.5 if off == 0 else 0.5
    init.u = u
    return tx, init


def _rand(rng, n, gc):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return list(rng.choice(list("ACGT"), n, p=p))


def make_locus(rng, flank=1500, kind=None, gene_len=(1500, 2300)):
    kind = kind or ("tata" if rng.random() < 0.5 else "gc")
    s = _rand(rng, flank, 0.42)
    pgc = rng.uniform(0.3, 0.45) if kind == "tata" else rng.uniform(0.65, 0.75)
    s[flank - 150:flank] = _rand(rng, 150, pgc)
    tss = flank
    G = int(rng.integers(*gene_len))
    s += _rand(rng, 60, pgc) + _rand(rng, G + flank - 60, 0.42)
    if kind == "tata":
        s[tss - 31:tss - 25] = "TATAAA"
        s[tss - 2:tss + 5] = "TCAGTTC"
    else:
        s[tss - 1:tss + 1] = "CA"
    donors, acceptors = [], []
    pos = tss + int(rng.integers(60, 200))
    for _ in range(int(rng.integers(1, 4))):
        ilen = int(rng.integers(250, 700))
        if pos + ilen + 250 > tss + G:
            break
        d, a = pos, pos + ilen - 1
        don = list("CAGGTAAGT")
        for k in (0, 1, 5, 6, 7, 8):
            if rng.random() < 0.15:
                don[k] = "ACGT"[rng.integers(4)]
        s[d - 3:d + 6] = don
        ppt = ["T" if rng.random() < 0.7 else "C" for _ in range(15)]
        s[a - 19:a - 4] = ppt
        s[a - 4:a + 2] = list("TTCAGG")
        donors.append(d); acceptors.append(a)
        pos = a + 1 + int(rng.integers(80, 220))

    def free(q, w):
        return not any(d - 12 <= q + w and q <= d + 12 or a - 25 <= q + w and q <= a + 4
                       for d, a in zip(donors, acceptors))
    for motif, n in (("AATAAA", int(rng.integers(2, 5))), ("TTTTTTT", int(rng.integers(1, 4)))):
        for _ in range(n):
            q = tss + int(rng.integers(80, G))
            if free(q, len(motif)):
                s[q:q + len(motif)] = motif
    return dict(seq="".join(s), tss=tss, gene_end=tss + G, donors=donors, acceptors=acceptors,
                promoter_gc=pgc, promoter_kind=kind)


def simulate(outdir, n_loci=200, reads_per_locus=40, seed=0, tss_window=100, verbose=True, truth=None,
             gene_len=(1500, 2300)):
    """truth: optional (TxHMM, InitModel or None) to simulate from instead of truth_models().
    Chain truth: caps from the initiation propensity in [tss - tss_window, tss + tss_window];
    legacy truth: caps from the InitModel PWM."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tx, init = truth or truth_models()
    chain = tx.cfg.chain
    loci = [make_locus(rng, gene_len=gene_len) for _ in range(n_loci)]
    store = SequenceStore([L["seq"] for L in loci])

    fasta_recs, bed_rows, bam_recs, truth_reads = [], [], [], []
    refs = []
    for li, L in enumerate(loci):
        chrom = f"chr{li + 1}"
        strand = "+" if rng.random() < 0.5 else "-"
        seq = L["seq"]
        n = len(seq)
        fasta_recs.append((chrom, seq if strand == "+" else revcomp(seq)))
        refs.append((chrom, n))
        to_g = (lambda x: x) if strand == "+" else (lambda x, n=n: n - 1 - x)
        L.update(chrom=chrom, strand=strand)
        cand = np.arange(L["tss"] - tss_window, L["tss"] + tss_window + 1)
        if chain:
            pin = tx.initiation_scan(store, store.starts[li] + cand)          # (n, classes)
            pr = (pin / pin.sum()).ravel()
            pick = rng.choice(len(pr), reads_per_locus, p=pr)
            tss, cls = cand[pick // pin.shape[1]], pick % pin.shape[1]
            start = tx.lay.e0_states[cls]
            L["initiation_profile_peak"] = int(cand[np.argmax(pin.sum(1))])
        else:
            eta = init.score(store, store.starts[li] + cand)
            pr = np.exp(eta - eta.max()); pr /= pr.sum()
            tss, cls, start = rng.choice(cand, reads_per_locus, p=pr), np.full(reads_per_locus, -1), None
        rs = ReadSet.transcripts(store, store.starts[li] + tss, L["gene_end"] - tss)
        smp = tx.sample(rs, rng, start_state=start)
        for r in range(reads_per_locus):
            t0, t1 = int(tss[r]), int(tss[r] + smp["end"][r])
            cl = "term" if smp["terminated"][r] else "surv"
            if smp["donor"][r] >= 0:
                d, a = t0 + int(smp["donor"][r]), t0 + int(smp["acceptor"][r])
                local_blocks = [(t0, d), (a + 1, t1 + 1)]
            else:
                d = a = -1
                local_blocks = [(t0, t1 + 1)]
            gblocks = sorted((min(to_g(x0), to_g(x1 - 1)), max(to_g(x0), to_g(x1 - 1)) + 1)
                             for x0, x1 in local_blocks)
            name = f"L{li}_r{r}|end={cl}"
            bed_rows.append((chrom, strand, gblocks, name))
            cig, prev = [], None
            for bs, be in gblocks:
                if prev is not None:
                    cig.append(("N", bs - prev))
                cig.append(("M", be - bs)); prev = be
            bam_recs.append(dict(name=name, ref=chrom, pos=gblocks[0][0],
                                 flag=16 if strand == "-" else 0, cigar=cig, mapq=60))
            cv = int(smp["cleave"][r])
            truth_reads.append(dict(locus=li, tss=to_g(t0), tss_rel=t0 - L["tss"], end=to_g(t1),
                                    strand=strand, donor=to_g(d) if d >= 0 else -1,
                                    acceptor=to_g(a) if a >= 0 else -1,
                                    track=int(smp["track"][r]), promoter_class=int(cls[r]),
                                    end_class=cl, cleaved=bool(cv >= 0 and smp["terminated"][r]),
                                    cleave_entry_to_end=(int(smp["end"][r]) - cv) if cv >= 0 else -1))
    write_fasta(os.path.join(outdir, "genome.fa"), fasta_recs)
    write_bed12(os.path.join(outdir, "reads.bed12"), bed_rows)
    write_bam(os.path.join(outdir, "reads.bam"), refs, bam_recs)
    meta = dict(loci=[{k: v for k, v in L.items() if k != "seq"} for L in loci], reads=truth_reads)
    with open(os.path.join(outdir, "truth.json"), "w") as fh:
        json.dump(meta, fh)
    if verbose:
        tr = np.array([r["track"] for r in truth_reads])
        spl = np.array([r["donor"] >= 0 for r in truth_reads])
        term = np.array([r["end_class"] == "term" for r in truth_reads])
        clv = np.array([r["cleaved"] for r in truth_reads])
        print(f"[sim] {n_loci} loci, {len(truth_reads)} reads; A-track {np.mean(tr == 0):.2f}, "
              f"spliced {spl.mean():.2f}, terminated {term.mean():.2f} (via cleavage {clv.mean():.2f})")
    return tx, init, meta
