"""Sequence store and sequence-only features.

Every locus is stored ONCE, oriented in the direction of transcription, in one
global uint8 array (A=0 C=1 G=2 T=3 N=4), separated by runs of N. A read (or a
predicted transcript) is then just a start index into that array plus a length,
so features are computed on the fly for the positions a batch needs.

All features are functions of the reference sequence only, never of where a
read ends -- that is what lets the model predict from DNA alone.
"""
import re
from dataclasses import dataclass

import numpy as np

from .genome import encode, BASES

SEP = 256  # N-padding between loci


@dataclass(frozen=True)
class FeatureDef:
    """A window feature: `kind` of `track` over [t+lo, t+hi] (inclusive).

    track : 'GC', 'A', 'C', 'G', 'T', 'AT', or 'motif:<regex>' (a motif counts at
            the position of its LAST nucleotide, so lo..hi < 0 means 'fully
            transcribed upstream of t')
    kind  : 'mean' (fraction), 'any' (0/1), 'count'
    """
    name: str
    track: str
    lo: int
    hi: int
    kind: str = "mean"


DEFAULT_HAZARD_FEATURES = (
    FeatureDef("pas[-35,-10]", "motif:A[AT]TAAA", -35, -10, "any"),
    FeatureDef("gc[-40,-1]", "GC", -40, -1, "mean"),
    FeatureDef("gc[0,40]", "GC", 0, 40, "mean"),
    FeatureDef("T[-8,0]", "T", -8, 0, "mean"),
)

DEFAULT_PROMOTER_FEATURES = (
    FeatureDef("prom_gc[-100,-1]", "GC", -100, -1, "mean"),
    FeatureDef("prom_gc[0,99]", "GC", 0, 99, "mean"),
    FeatureDef("tata[-35,-20]", "motif:TATA[AT]A", -35, -20, "any"),
)

DEFAULT_INIT_FEATURES = (
    FeatureDef("gc[-100,-1]", "GC", -100, -1, "mean"),
    FeatureDef("gc[0,99]", "GC", 0, 99, "mean"),
)


class SequenceStore:
    """Concatenated, oriented locus sequences with cached cumulative tracks."""

    def __init__(self, seqs):
        parts, starts, pos = [np.full(SEP, 4, np.uint8)], [], SEP
        for s in seqs:
            codes = encode(s) if isinstance(s, str) else np.asarray(s, np.uint8)
            starts.append(pos)
            parts.append(codes)
            parts.append(np.full(SEP, 4, np.uint8))
            pos += len(codes) + SEP
        self.bases = np.concatenate(parts)
        self.starts = np.array(starts, np.int64)
        self.lengths = np.array([len(s) for s in seqs], np.int64)
        self._cum = {}
        self._str = None

    # ---- basic access ------------------------------------------------------

    def __len__(self):
        return len(self.bases)

    def locus_seq(self, i):
        s = self.starts[i]
        return self._string()[s:s + self.lengths[i]]

    def gather(self, g):
        """bases at global positions g (any shape); out of range -> N."""
        g = np.asarray(g)
        ok = (g >= 0) & (g < len(self.bases))
        return np.where(ok, self.bases[np.clip(g, 0, len(self.bases) - 1)], 4)

    # ---- cumulative tracks -------------------------------------------------

    def _string(self):
        if self._str is None:
            table = bytes.maketrans(b"\x00\x01\x02\x03\x04", b"ACGTN")
            self._str = self.bases.tobytes().translate(table).decode()
        return self._str

    def cumtrack(self, track):
        if track not in self._cum:
            b = self.bases
            if track == "GC":
                ind = (b == 1) | (b == 2)
            elif track == "AT":
                ind = (b == 0) | (b == 3)
            elif track in ("A", "C", "G", "T"):
                ind = b == "ACGT".index(track)
            elif track.startswith("motif:"):
                ind = np.zeros(len(b), bool)
                pat = re.compile(f"(?=({track[6:]}))")
                for m in pat.finditer(self._string()):
                    ind[m.end(1) - 1] = True
            else:
                raise ValueError(f"unknown track {track!r}")
            self._cum[track] = np.concatenate([[0], np.cumsum(ind, dtype=np.int32)])
        return self._cum[track]

    def window(self, fd: FeatureDef, g):
        cs = self.cumtrack(fd.track)
        n = len(self.bases)
        a = np.clip(np.asarray(g) + fd.lo, 0, n)
        b = np.clip(np.asarray(g) + fd.hi + 1, 0, n)
        cnt = (cs[b] - cs[np.minimum(a, b)]).astype(float)
        if fd.kind == "mean":
            return cnt / (fd.hi - fd.lo + 1)
        if fd.kind == "any":
            return np.minimum(cnt, 1.0)
        return cnt

    def features(self, fdefs, g):
        """(..., F) matrix of window features at positions g."""
        if not fdefs:
            return np.zeros(np.shape(g) + (0,))
        return np.stack([self.window(fd, g) for fd in fdefs], axis=-1)

    # ---- k-mers, PWM windows, dinucleotide gates ---------------------------

    def kmer_index(self, g, k):
        """Index of the k-mer ENDING at g (4**k = unknown / contains N)."""
        g = np.asarray(g)
        code = np.zeros(g.shape, np.int64)
        bad = np.zeros(g.shape, bool)
        for j in range(k):
            b = self.gather(g - (k - 1 - j))
            bad |= b > 3
            code = code * 4 + np.minimum(b, 3)
        return np.where(bad, 4 ** k, code)

    def window_bases(self, g, offsets):
        """(n, K) base codes at g[:,None] + offsets."""
        return self.gather(np.asarray(g)[:, None] + np.asarray(offsets)[None, :])

    def dinuc_mask(self, g, first_offset, allowed):
        """True where bases[g+off], bases[g+off+1] form one of `allowed`."""
        b0 = self.gather(np.asarray(g) + first_offset)
        b1 = self.gather(np.asarray(g) + first_offset + 1)
        ok = np.zeros(np.shape(g), bool)
        for d in allowed:
            ok |= (b0 == "ACGT".index(d[0])) & (b1 == "ACGT".index(d[1]))
        return ok


def kmer_names(k):
    names = []
    for i in range(4 ** k):
        s, x = "", i
        for _ in range(k):
            s = "ACGT"[x % 4] + s
            x //= 4
        names.append(s)
    return names + ["UNK"]
