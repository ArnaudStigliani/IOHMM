"""Reads (BAM or BED12) + genome FASTA  ->  TxDataset.

No annotation is used: the TSS comes from each read's 5' end (cap), introns
from its junctions, the strand from the alignment, and loci are built by
merging overlapping reads on the same strand.

Per read we keep, in LOCUS coordinates (oriented 5'->3' along transcription):
  tss, end (inclusive), end type, weight, and its junctions
  (donor = first intron nt, acceptor = last intron nt), snapped to canonical
  dinucleotides when the aligner placed them a few nt off.
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np

from .genome import FastaFile, iter_bam, iter_bed12, revcomp, encode
from .sequence import SequenceStore

END_TYPES = {"terminated": 0, "censored": 1, "mixture": 2, "survived": 3}
END_NAMES = {v: k for k, v in END_TYPES.items()}


@dataclass
class PrepConfig:
    min_mapq: int = 0
    skip_secondary: bool = True
    min_intron: int = 20          # CIGAR N shorter than this is not an intron
    strand_mode: str = "read"     # 'read' | 'reverse' | 'tag:XS' | 'ts'
    cap_mode: str = "all"         # 'all' | 'tag:NAME' | 'name:REGEX'
    end_mode: str = "none"        # 'none' | 'tag:NAME:THRESH' | 'softclip:MINLEN:MINFRAC' | 'name:REGEX'
    end_types: dict = field(default_factory=lambda: {"pA-": "mixture", "pA+": "terminated"})
    weight_tag: str = None        # optional numeric tag holding a per-read weight
    flank: int = 1000             # sequence kept on each side of a locus
    snap: int = 5                 # max shift (nt) when snapping junctions
    donor_dinucs: tuple = ("GT", "GC")
    acceptor_dinucs: tuple = ("AG",)
    min_len: int = 20
    max_len: int = 50000          # longer reads are cut and marked 'survived'
    locus_gap: int = 0


# ------------------------------------------------------------ read parsing

def _strand(aln, mode):
    base = "-" if aln.reverse else "+"
    if mode == "read":
        return base
    if mode == "reverse":
        return "+" if base == "-" else "-"
    if mode == "ts":
        ts = aln.tags.get("ts")
        if ts is None:
            return None
        return base if ts == "+" else ("+" if base == "-" else "-")
    if mode.startswith("tag:"):
        v = aln.tags.get(mode[4:])
        return v if v in ("+", "-") else None
    raise ValueError(f"unknown strand_mode {mode!r}")


def _truthy(v):
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "no", "false", "f", "n")
    return bool(v)


def _capped(aln, mode):
    if mode == "all":
        return True
    if mode.startswith("tag:"):
        return _truthy(aln.tags.get(mode[4:]))
    if mode.startswith("name:"):
        return re.search(mode[5:], aln.name) is not None
    raise ValueError(f"unknown cap_mode {mode!r}")


def _end_class(aln, strand, mode):
    if mode == "none":
        return "pA-"
    if mode.startswith("tag:"):
        _, name, thr = mode.split(":")
        v = aln.tags.get(name)
        return "pA+" if v is not None and float(v) >= float(thr) else "pA-"
    if mode.startswith("softclip:"):
        _, minlen, minfrac = mode.split(":")
        clip = aln.clip_right if strand == "+" else aln.clip_left
        base = "A" if strand == "+" else "T"
        if len(clip) >= int(minlen) and clip.count(base) / len(clip) >= float(minfrac):
            return "pA+"
        return "pA-"
    if mode.startswith("name:"):
        m = re.search(mode[5:], aln.name)
        return m.group(1) if m else "pA-"
    raise ValueError(f"unknown end_mode {mode!r}")


def parse_alignments(alignments, cfg: PrepConfig):
    """-> list of dicts in GENOMIC coordinates, plus filter statistics."""
    stats = defaultdict(int)
    reads = []
    for aln in alignments:
        stats["total"] += 1
        if cfg.skip_secondary and aln.flag & (256 | 2048):
            stats["secondary/supplementary"] += 1
            continue
        if aln.mapq < cfg.min_mapq:
            stats["low_mapq"] += 1
            continue
        strand = aln.strand if aln.strand in ("+", "-") and cfg.strand_mode == "read" \
            else _strand(aln, cfg.strand_mode)
        if strand is None:
            stats["no_strand"] += 1
            continue
        capped = _capped(aln, cfg.cap_mode)
        blocks = sorted(aln.blocks)
        s, e = blocks[0][0], blocks[-1][1]
        tss = s if strand == "+" else e - 1
        end3 = e - 1 if strand == "+" else s
        introns = [(blocks[i][1], blocks[i + 1][0]) for i in range(len(blocks) - 1)]
        cls = _end_class(aln, strand, cfg.end_mode)
        etype = cfg.end_types.get(cls, "drop")
        w = float(aln.tags.get(cfg.weight_tag, 1.0)) if cfg.weight_tag else 1.0
        reads.append(dict(name=aln.name, chrom=aln.chrom, strand=strand, start=s, end=e,
                          tss=tss, end3=end3, introns=introns, capped=capped,
                          end_class=cls, end_type=etype, weight=w))
    return reads, stats


# ------------------------------------------------------------------ loci

def build_loci(reads, gap=0):
    """Merge overlapping read spans per (chrom, strand)."""
    by = defaultdict(list)
    for i, r in enumerate(reads):
        by[(r["chrom"], r["strand"])].append(i)
    loci, read_locus = [], np.full(len(reads), -1, int)
    for (chrom, strand), idx in sorted(by.items()):
        idx.sort(key=lambda i: reads[i]["start"])
        cur, cs, ce = [], None, None
        for i in idx:
            s, e = reads[i]["start"], reads[i]["end"]
            if cur and s <= ce + gap:
                cur.append(i); ce = max(ce, e)
            else:
                if cur:
                    loci.append((chrom, strand, cs, ce, cur))
                cur, cs, ce = [i], s, e
        if cur:
            loci.append((chrom, strand, cs, ce, cur))
    for li, (_, _, _, _, members) in enumerate(loci):
        read_locus[members] = li
    return loci, read_locus


# --------------------------------------------------------------- dataset

class TxDataset:
    """Reads in locus coordinates + the oriented locus sequences.

    reads: dict of arrays (one entry per read)
      locus, tss, end (inclusive, locus coords), end_type, weight,
      jptr (junctions of read i are jdon[jptr[i]:jptr[i+1]] etc.)
    caps : locus, pos  -- every capped read's TSS (for the initiation model)
    kin  : optional known introns per locus (txhmm structures): ptr (n_loci + 1), don, acc --
           introns of locus l are don[ptr[l]:ptr[l+1]] (first intron nt) .. acc[...] (last
           intron nt), oriented locus coordinates, sorted, non-overlapping
    """

    def __init__(self, store, loci, reads, jdon, jacc, jcanon, caps, meta=None, kin=None):
        self.store, self.loci, self.reads = store, loci, reads
        self.jdon, self.jacc, self.jcanon = jdon, jacc, jcanon
        self.caps = caps
        self.meta = meta or {}
        self.kin = kin

    def __len__(self):
        return len(self.reads["tss"])

    @property
    def lengths(self):
        return self.reads["end"] - self.reads["tss"] + 1

    def gstart(self, idx=None):
        """Global store index of each read's TSS."""
        r = self.reads if idx is None else {k: v[idx] for k, v in self.reads.items()}
        return self.store.starts[r["locus"]] + r["tss"]

    def junctions(self, i):
        a, b = self.reads["jptr"][i], self.reads["jptr"][i + 1]
        return list(zip(self.jdon[a:b], self.jacc[a:b], self.jcanon[a:b]))

    def first_junction(self):
        """Per read: (donor_rel, acceptor_rel, canonical) of the 5'-most junction, -1 if none."""
        n = len(self)
        d = np.full(n, -1); a = np.full(n, -1); c = np.zeros(n, bool)
        ptr = self.reads["jptr"]
        has = ptr[1:] > ptr[:-1]
        first = ptr[:-1][has]
        d[has] = self.jdon[first] - self.reads["tss"][has]
        a[has] = self.jacc[first] - self.reads["tss"][has]
        c[has] = self.jcanon[first]
        return d, a, c

    def subset_loci(self, locus_ids):
        keep_l = np.zeros(len(self.loci["chrom"]), bool)
        keep_l[np.asarray(locus_ids)] = True
        keep = keep_l[self.reads["locus"]]
        idx = np.nonzero(keep)[0]
        ptr = self.reads["jptr"]
        counts = ptr[idx + 1] - ptr[idx]
        new_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        jsel = np.repeat(ptr[idx] - new_ptr[:-1], counts) + np.arange(new_ptr[-1])   # junction rows kept
        reads = {k: v[idx] for k, v in self.reads.items() if k != "jptr"}
        reads["jptr"] = new_ptr
        ck = keep_l[self.caps["locus"]]
        caps = {k: v[ck] for k, v in self.caps.items()}
        return TxDataset(self.store, self.loci, reads, self.jdon[jsel], self.jacc[jsel],
                         self.jcanon[jsel], caps, dict(self.meta), self.kin)

    def split(self, test_frac=0.2, seed=0):
        """Split BY LOCUS into (train, test)."""
        nl = len(self.loci["chrom"])
        used = np.unique(self.reads["locus"])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(used)
        nt = max(1, int(round(test_frac * len(perm))))
        return self.subset_loci(np.sort(perm[nt:])), self.subset_loci(np.sort(perm[:nt]))

    # ---- persistence -------------------------------------------------------

    def save(self, path):
        np.savez_compressed(
            path, bases=self.store.bases, starts=self.store.starts, lengths=self.store.lengths,
            **{f"locus_{k}": np.asarray(v) for k, v in self.loci.items()},
            **{f"read_{k}": v for k, v in self.reads.items()},
            **{f"cap_{k}": v for k, v in self.caps.items()},
            jdon=self.jdon, jacc=self.jacc, jcanon=self.jcanon,
            **({f"kin_{k}": v for k, v in self.kin.items()} if self.kin is not None else {}),
            meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        store = SequenceStore.__new__(SequenceStore)
        store.bases, store.starts, store.lengths = z["bases"], z["starts"], z["lengths"]
        store._cum, store._str = {}, None
        pick = lambda p: {k[len(p):]: z[k] for k in z.files if k.startswith(p)}
        meta = json.loads(str(z["meta"]))
        kin = pick("kin_") or None
        return cls(store, pick("locus_"), pick("read_"), z["jdon"], z["jacc"], z["jcanon"],
                   pick("cap_"), meta, kin)

    def summary(self):
        L = self.lengths
        ptr = self.reads["jptr"]
        nj = ptr[1:] - ptr[:-1]
        et = self.reads["end_type"]
        return {
            "loci": int(len(np.unique(self.reads["locus"]))),
            "reads": int(len(self)),
            "median_length": float(np.median(L)) if len(L) else 0.0,
            "spliced_reads": int((nj > 0).sum()),
            "junctions": int(nj.sum()),
            "canonical_junction_frac": float(self.jcanon.mean()) if len(self.jcanon) else float("nan"),
            "end_types": {END_NAMES[k]: int((et == k).sum()) for k in np.unique(et)},
            "caps": int(len(self.caps["pos"])),
        }


def _snap(codes, d, a, snap, donors, acceptors):
    """Shift a junction (same shift at both ends) onto canonical dinucleotides."""
    def ok(dd, aa):
        if dd < 0 or aa < 1 or dd + 1 >= len(codes) or aa >= len(codes):
            return False
        dn = "ACGTN"[codes[dd]] + "ACGTN"[codes[dd + 1]]
        an = "ACGTN"[codes[aa - 1]] + "ACGTN"[codes[aa]]
        return dn in donors and an in acceptors
    for delta in sorted(range(-snap, snap + 1), key=abs):
        if ok(d + delta, a + delta):
            return d + delta, a + delta, True
    return d, a, False


def build_dataset(alignments, fasta_path, cfg: PrepConfig = None, verbose=True):
    cfg = cfg or PrepConfig()
    fasta = FastaFile(fasta_path)
    raw, stats = parse_alignments(alignments, cfg)
    loci, read_locus = build_loci(raw, cfg.locus_gap)

    seqs, L_chrom, L_strand, L_g0, L_g1 = [], [], [], [], []
    for chrom, strand, s, e, _ in loci:
        g0, g1 = s - cfg.flank, e + cfg.flank
        seq = fasta.fetch(chrom, g0, g1)
        seqs.append(revcomp(seq) if strand == "-" else seq)
        L_chrom.append(chrom); L_strand.append(strand); L_g0.append(g0); L_g1.append(g1)
    store = SequenceStore(seqs)
    codes_by_locus = [encode(s) for s in seqs]

    def to_local(li, x):
        return x - L_g0[li] if L_strand[li] == "+" else (L_g1[li] - 1) - x

    cols = defaultdict(list)
    jd, ja, jc, jptr = [], [], [], [0]
    cap_l, cap_p = [], []
    for i, r in enumerate(raw):
        li = read_locus[i]
        tss, end = to_local(li, r["tss"]), to_local(li, r["end3"])
        if r["capped"]:
            cap_l.append(li); cap_p.append(tss)
        if not r["capped"]:
            stats["uncapped"] += 1
            continue
        if r["end_type"] == "drop":
            stats[f"end_class_dropped:{r['end_class']}"] += 1
            continue
        etype = r["end_type"]
        if end - tss + 1 < cfg.min_len:
            stats["too_short"] += 1
            continue
        cut = False
        if end - tss + 1 > cfg.max_len:
            end, etype, cut = tss + cfg.max_len - 1, "survived", True
            stats["cut_at_max_len"] += 1
        juncs = []
        for (s_, e_) in r["introns"]:
            if r["strand"] == "+":
                d, a = to_local(li, s_), to_local(li, e_ - 1)
            else:
                d, a = to_local(li, e_ - 1), to_local(li, s_)
            d, a, canon = _snap(codes_by_locus[li], d, a, cfg.snap,
                                cfg.donor_dinucs, cfg.acceptor_dinucs)
            # a junction crossing the cut stays (it ends past the read): the kept stretch is inside that intron
            if tss < d and (a < end or (cut and d <= end)):
                juncs.append((d, a, canon))
        juncs.sort()
        for d, a, c in juncs:
            jd.append(d); ja.append(a); jc.append(c)
        jptr.append(len(jd))
        cols["locus"].append(li); cols["tss"].append(tss); cols["end"].append(end)
        cols["end_type"].append(END_TYPES[etype]); cols["weight"].append(r["weight"])
        cols["cut"].append(cut)
        stats["kept"] += 1

    reads = {k: np.array(v) for k, v in cols.items()}
    reads["locus"] = reads["locus"].astype(np.int64)
    reads["jptr"] = np.array(jptr, np.int64)
    loci_tab = {"chrom": np.array(L_chrom), "strand": np.array(L_strand),
                "g0": np.array(L_g0, np.int64), "g1": np.array(L_g1, np.int64),
                "flank": np.full(len(L_g0), cfg.flank, np.int64)}
    caps = {"locus": np.array(cap_l, np.int64), "pos": np.array(cap_p, np.int64)}
    meta = {"prep": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
            "stats": dict(stats)}
    ds = TxDataset(store, loci_tab, reads, np.array(jd, np.int64), np.array(ja, np.int64),
                   np.array(jc, bool), caps, meta)
    if verbose:
        print("[prep] filters:", dict(stats))
        print("[prep] dataset:", ds.summary())
    return ds


def dataset_from_bam(bam_path, fasta_path, cfg: PrepConfig = None, verbose=True):
    cfg = cfg or PrepConfig()
    need_seq = cfg.end_mode.startswith("softclip")
    return build_dataset(iter_bam(bam_path, cfg.min_intron, need_seq), fasta_path, cfg, verbose)


def dataset_from_bed12(bed_path, fasta_path, cfg: PrepConfig = None, verbose=True):
    return build_dataset(iter_bed12(bed_path), fasta_path, cfg, verbose)


def locus_to_genomic(ds, locus, pos):
    """Map locus coordinates back to genomic (0-based)."""
    g0, g1, strand = ds.loci["g0"][locus], ds.loci["g1"][locus], ds.loci["strand"][locus]
    return g0 + pos if strand == "+" else (g1 - 1) - pos
