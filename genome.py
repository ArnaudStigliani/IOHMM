"""Genome and alignment I/O.

* FastaFile   : random access to an (optionally soft-masked) FASTA via a .fai
                index, which is created if missing. No external dependency.
* iter_bam    : BAM records. Uses pysam when installed, otherwise a small
                pure-Python BAM reader (sequential, no index needed).
* write_bam   : minimal BGZF/BAM writer (used by the simulator and tests).
* iter_bed12  : BED12 records.

All coordinates are 0-based, half-open, as in BAM/BED.
"""
import gzip
import os
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------ nucleotides

BASES = "ACGTN"
_ENC = np.full(256, 4, np.uint8)
for _i, _c in enumerate("ACGT"):
    _ENC[ord(_c)] = _i
    _ENC[ord(_c.lower())] = _i
_ENC[ord("U")] = _ENC[ord("u")] = 3
_COMP = str.maketrans("ACGTNacgtnRYKMSWBDHVrykmswbdhv",
                      "TGCANtgcanYRMKSWVHDByrmkswvhdb")


def encode(seq: str) -> np.ndarray:
    """ACGT -> 0..3, anything else -> 4 (N)."""
    return _ENC[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]


def decode(codes) -> str:
    return "".join(BASES[c] for c in codes)


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ------------------------------------------------------------------ FASTA

class FastaFile:
    """Indexed FASTA reader (samtools-compatible .fai)."""

    def __init__(self, path):
        self.path = path
        fai = path + ".fai"
        if os.path.exists(fai):
            self.index = self._read_fai(fai)
        else:
            self.index = self._build_index(path)
            try:
                with open(fai, "w") as fh:
                    for name, (ln, off, lb, lw) in self.index.items():
                        fh.write(f"{name}\t{ln}\t{off}\t{lb}\t{lw}\n")
            except OSError:
                pass
        self._fh = open(path, "rb")

    @staticmethod
    def _read_fai(fai):
        idx = {}
        with open(fai) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
        return idx

    @staticmethod
    def _build_index(path):
        idx, name, length, offset, lb, lw = {}, None, 0, 0, 0, 0
        pos = 0
        with open(path, "rb") as fh:
            for line in fh:
                if line.startswith(b">"):
                    if name is not None:
                        idx[name] = (length, offset, lb, lw)
                    name = line[1:].split()[0].decode()
                    length, lb, lw = 0, 0, 0
                    offset = pos + len(line)
                else:
                    stripped = line.rstrip(b"\r\n")
                    if lb == 0:
                        lb, lw = len(stripped), len(line)
                    length += len(stripped)
                pos += len(line)
        if name is not None:
            idx[name] = (length, offset, lb, lw)
        return idx

    @property
    def references(self):
        return list(self.index)

    def get_reference_length(self, chrom):
        return self.index[chrom][0]

    def fetch(self, chrom, start, end):
        """Upper-case sequence of chrom[start:end]; out-of-range is padded with N."""
        length, off, lb, lw = self.index[chrom]
        s, e = max(start, 0), min(end, length)
        if e <= s:
            return "N" * max(end - start, 0)
        b0 = off + (s // lb) * lw + s % lb
        b1 = off + (e // lb) * lw + e % lb
        self._fh.seek(b0)
        raw = self._fh.read(b1 - b0).replace(b"\n", b"").replace(b"\r", b"")
        seq = raw.decode().upper()
        return "N" * (s - start) + seq + "N" * (end - e)


def write_fasta(path, records, width=60):
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")
    if os.path.exists(path + ".fai"):
        os.remove(path + ".fai")


# ------------------------------------------------------------- alignments

@dataclass
class Alignment:
    """One aligned read, reduced to what the pipeline needs."""
    name: str
    chrom: str
    blocks: list          # [(start, end)] genomic, split at introns (CIGAR N)
    reverse: bool         # read aligned to the minus strand
    mapq: int = 255
    flag: int = 0
    tags: dict = field(default_factory=dict)
    clip_left: str = ""   # soft-clipped query bases, left / right in genome order
    clip_right: str = ""
    strand: str = None    # set directly for BED12


_CIGAR_OPS = "MIDNSHP=X"


def blocks_from_cigar(start, cigar, min_intron=20):
    """cigar: [(op_char, length)] -> aligned blocks split at N >= min_intron."""
    blocks, cur_s, pos = [], start, start
    for op, ln in cigar:
        if op in "M=XD":
            pos += ln
        elif op == "N":
            if ln >= min_intron:
                blocks.append((cur_s, pos))
                pos += ln
                cur_s = pos
            else:
                pos += ln
    blocks.append((cur_s, pos))
    return [b for b in blocks if b[1] > b[0]]


def _soft_clips(cigar, seq):
    left = right = ""
    if seq and cigar:
        if cigar[0][0] == "S":
            left = seq[:cigar[0][1]]
        if cigar[-1][0] == "S":
            right = seq[len(seq) - cigar[-1][1]:]
    return left, right


def iter_bam(path, min_intron=20, need_seq=False):
    """Yield Alignment objects from a BAM (pysam if available)."""
    try:
        import pysam  # noqa: F401
        yield from _iter_bam_pysam(path, min_intron, need_seq)
    except ImportError:
        yield from _iter_bam_python(path, min_intron, need_seq)


def _iter_bam_pysam(path, min_intron, need_seq):
    import pysam
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped:
                continue
            cig = [(_CIGAR_OPS[o], n) for o, n in r.cigartuples]
            seq = r.query_sequence if need_seq else None
            cl, cr = _soft_clips(cig, seq)
            yield Alignment(r.query_name, r.reference_name,
                            blocks_from_cigar(r.reference_start, cig, min_intron),
                            r.is_reverse, r.mapping_quality, r.flag,
                            dict(r.get_tags()), cl, cr)


_SEQ_NT16 = "=ACMGRSVTWYHKDBN"
_TAG_FMT = {"c": "<b", "C": "<B", "s": "<h", "S": "<H", "i": "<i", "I": "<I", "f": "<f"}


def _parse_tags(buf):
    tags, i, n = {}, 0, len(buf)
    while i < n:
        key = buf[i:i + 2].decode()
        t = chr(buf[i + 2])
        i += 3
        if t == "A":
            tags[key] = chr(buf[i]); i += 1
        elif t in _TAG_FMT:
            fmt = _TAG_FMT[t]
            tags[key] = struct.unpack_from(fmt, buf, i)[0]
            i += struct.calcsize(fmt)
        elif t in "ZH":
            j = buf.index(b"\0", i)
            tags[key] = buf[i:j].decode()
            i = j + 1
        elif t == "B":
            sub = chr(buf[i]); cnt = struct.unpack_from("<i", buf, i + 1)[0]
            fmt = _TAG_FMT[sub]
            sz = struct.calcsize(fmt)
            tags[key] = list(struct.unpack_from("<" + fmt[1] * cnt, buf, i + 5))
            i += 5 + sz * cnt
        else:
            raise ValueError(f"unknown BAM tag type {t!r}")
    return tags


def _iter_bam_python(path, min_intron, need_seq):
    with gzip.open(path, "rb") as fh:
        if fh.read(4) != b"BAM\x01":
            raise ValueError(f"{path} is not a BAM file")
        l_text = struct.unpack("<i", fh.read(4))[0]
        fh.read(l_text)
        n_ref = struct.unpack("<i", fh.read(4))[0]
        refs = []
        for _ in range(n_ref):
            ln = struct.unpack("<i", fh.read(4))[0]
            refs.append(fh.read(ln)[:-1].decode())
            fh.read(4)
        while True:
            head = fh.read(4)
            if len(head) < 4:
                break
            bs = struct.unpack("<i", head)[0]
            rec = fh.read(bs)
            (ref_id, pos, l_name, mapq, _bin, n_cig, flag, l_seq,
             _nr, _np, _tl) = struct.unpack_from("<iiBBHHHIiii", rec, 0)
            if ref_id < 0 or flag & 4:
                continue
            o = 32
            name = rec[o:o + l_name - 1].decode(); o += l_name
            cig_raw = struct.unpack_from("<" + "I" * n_cig, rec, o); o += 4 * n_cig
            cig = [(_CIGAR_OPS[v & 0xF], v >> 4) for v in cig_raw]
            seq = None
            nb = (l_seq + 1) // 2
            if need_seq:
                sb = rec[o:o + nb]
                seq = "".join(_SEQ_NT16[b >> 4] + _SEQ_NT16[b & 0xF] for b in sb)[:l_seq]
            o += nb + l_seq
            tags = _parse_tags(rec[o:]) if o < len(rec) else {}
            cl, cr = _soft_clips(cig, seq)
            yield Alignment(name, refs[ref_id], blocks_from_cigar(pos, cig, min_intron),
                            bool(flag & 16), mapq, flag, tags, cl, cr)


def _bgzf_block(data: bytes) -> bytes:
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    comp = c.compress(data) + c.flush()
    header = struct.pack("<BBBBIBBHBBHH", 31, 139, 8, 4, 0, 0, 255, 6,
                         66, 67, 2, len(comp) + 25)
    return header + comp + struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data))


_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


def write_bam(path, references, records):
    """references: [(name, length)]; records: dicts with
       name, ref, pos, flag, mapq, cigar [(op,len)], seq (str or ''), tags {k: (type, value)}"""
    rid = {n: i for i, (n, _) in enumerate(references)}
    out = bytearray()
    text = "@HD\tVN:1.6\tSO:unsorted\n" + "".join(f"@SQ\tSN:{n}\tLN:{l}\n" for n, l in references)
    out += b"BAM\x01" + struct.pack("<i", len(text)) + text.encode()
    out += struct.pack("<i", len(references))
    for n, l in references:
        nb = n.encode() + b"\0"
        out += struct.pack("<i", len(nb)) + nb + struct.pack("<i", l)
    for r in records:
        name = r["name"].encode() + b"\0"
        cig = [(_CIGAR_OPS.index(op), ln) for op, ln in r["cigar"]]
        seq = r.get("seq", "")
        l_seq = len(seq)
        body = bytearray()
        body += struct.pack("<iiBBHHHIiii", rid[r["ref"]], r["pos"], len(name), r.get("mapq", 60),
                            4680, len(cig), r.get("flag", 0), l_seq, -1, -1, 0)
        body += name
        body += struct.pack("<" + "I" * len(cig), *[(ln << 4) | op for op, ln in cig])
        if l_seq:
            codes = [_SEQ_NT16.index(c) if c in _SEQ_NT16 else 15 for c in seq]
            if l_seq % 2:
                codes.append(0)
            body += bytes((codes[i] << 4) | codes[i + 1] for i in range(0, len(codes), 2))
            body += b"\xff" * l_seq
        for k, (t, v) in r.get("tags", {}).items():
            body += k.encode() + t.encode()
            if t == "A":
                body += v.encode()
            elif t == "Z":
                body += v.encode() + b"\0"
            else:
                body += struct.pack(_TAG_FMT[t], v)
        out += struct.pack("<i", len(body)) + body
    with open(path, "wb") as fh:
        for i in range(0, len(out), 60000):
            fh.write(_bgzf_block(bytes(out[i:i + 60000])))
        fh.write(_BGZF_EOF)


def iter_bed12(path):
    """Yield Alignment objects from a BED12 file (strand from column 6)."""
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            chrom, s, e = p[0], int(p[1]), int(p[2])
            name = p[3] if len(p) > 3 else "."
            score = p[4] if len(p) > 4 else "0"
            strand = p[5] if len(p) > 5 else "+"
            if len(p) >= 12 and int(p[9]) > 0:
                sizes = [int(x) for x in p[10].rstrip(",").split(",")]
                starts = [int(x) for x in p[11].rstrip(",").split(",")]
                blocks = [(s + a, s + a + b) for a, b in zip(starts, sizes)]
            else:
                blocks = [(s, e)]
            tags = {"score": score}
            yield Alignment(name, chrom, blocks, strand == "-", 255, 0, tags, strand=strand)


def write_bed12(path, rows):
    """rows: (chrom, strand, blocks [(s,e)], name)."""
    with open(path, "w") as fh:
        for chrom, strand, blocks, name in rows:
            blocks = sorted(blocks)
            s, e = blocks[0][0], blocks[-1][1]
            sizes = ",".join(str(b - a) for a, b in blocks)
            starts = ",".join(str(a - s) for a, _ in blocks)
            fh.write(f"{chrom}\t{s}\t{e}\t{name}\t0\t{strand}\t{s}\t{e}\t0\t"
                     f"{len(blocks)}\t{sizes},\t{starts},\n")
