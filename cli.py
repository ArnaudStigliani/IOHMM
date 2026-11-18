"""Command line:  python -m txhmm {prep,fit,predict,inspect,simulate} ...

  prep      BAM/BED12 + FASTA -> dataset.npz
  fit       dataset.npz -> PREFIX.tx.npz (initiation, fate, splicing, termination)
  predict   model + FASTA + region (+ where Pol II sits) -> bedGraph tracks + PREFIX.tss.json
  inspect   model (+ dataset) -> top k-mers of every transition, distributions, PWMs
  simulate  synthetic genome.fa / reads.bed12 / reads.bam / truth.json
"""
import argparse
import json
import sys

import numpy as np


def _end_types(s):
    out = {}
    for item in s.split(","):
        k, v = item.split(":")
        out[k] = v
    return out


def cmd_prep(a):
    from .dataprep import PrepConfig, dataset_from_bam, dataset_from_bed12
    cfg = PrepConfig(min_mapq=a.min_mapq, min_intron=a.min_intron, strand_mode=a.strand_mode,
                     cap_mode=a.cap_mode, end_mode=a.end_mode, end_types=_end_types(a.end_types),
                     weight_tag=a.weight_tag, flank=a.flank, snap=a.snap, min_len=a.min_len,
                     max_len=a.max_len)
    ds = dataset_from_bam(a.bam, a.fasta, cfg) if a.bam else dataset_from_bed12(a.bed12, a.fasta, cfg)
    ds.save(a.out)
    print(f"[prep] saved {a.out}")


def _fit_init(train, test, pad, path):
    """Initiation model: independent of the fate model, so it can run alongside it."""
    import time
    from .model import InitModel
    t0 = time.perf_counter()
    im = InitModel(pad=pad).fit(train)
    if test is not None:
        print("[init] held-out:", im.evaluate(test), flush=True)
    im.save(path)
    print(f"[time] initiation model: {time.perf_counter() - t0:.1f} s", flush=True)


def cmd_fit(a):
    import time
    from .dataprep import TxDataset, END_TYPES
    from .model import TxConfig, TxHMM, ReadSet
    from .sequence import DEFAULT_HAZARD_FEATURES, DEFAULT_PROMOTER_FEATURES
    t0 = time.perf_counter()
    ds = TxDataset.load(a.data)
    print(f"[time] load dataset: {time.perf_counter() - t0:.1f} s "
          f"({len(ds)} reads, {len(ds.store.bases) / 1e6:.0f} Mb of locus sequence)", flush=True)
    min_len = a.min_len or int(ds.meta.get("prep", {}).get("min_len", 1))
    t0 = time.perf_counter()
    train, test = ds.split(a.test_frac, seed=a.seed) if a.test_frac > 0 else (ds, None)
    print(f"[time] split by locus: {time.perf_counter() - t0:.1f} s", flush=True)

    # legacy initiation: a separate initiation model, in its own process while the fate model trains
    init_proc = None
    if a.legacy_init:
        if a.workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
            init_proc = ctx.Process(target=_fit_init, args=(train, test, a.init_pad, a.out + ".init.npz"))
            init_proc.start()
        else:
            _fit_init(train, test, a.init_pad, a.out + ".init.npz")

    snapshot = a.snapshot
    if (ds.reads["end_type"] == END_TYPES["mixture"]).any() and not snapshot:
        print("[fit] dataset has 'mixture' ends -> enabling the snapshot (harvest) hazard")
        snapshot = True
    if a.kmer_transitions == "all":
        kt = ("donor", "acceptor", "abandon", "splice") + (("switch",) if a.switching else ()) \
             + (("commit",) if a.legacy_init else ())
    else:
        kt = tuple(x for x in a.kmer_transitions.split(",") if x)
    cfg = TxConfig(tracks=tuple(a.tracks.split(",")), switching=a.switching,
                   splicing=not a.no_splicing, n_wait=a.n_wait, min_intron=a.min_intron,
                   initiation="legacy" if a.legacy_init else "chain",
                   upstream=a.upstream, start_spread=a.start_spread, init_window_down=a.init_window_down, init_score=a.init_score,
                   promoter_classes=a.promoter_classes, promoter_substates=a.promoter_substates,
                   early_substates=a.early_substates, upstream_lookahead=a.upstream_lookahead,
                   class_track_lean=a.class_track_lean,
                   cleave_substates=a.cleave_substates, fix_intron_offsets=a.fix_intron_offsets, all_introns=a.all_introns or a.known_introns,
                   known_introns=a.known_introns,
                   freeze=tuple(x for x in a.freeze.split(",") if x),
                   hazard_features=DEFAULT_HAZARD_FEATURES if a.legacy_features else (),
                   promoter_features=DEFAULT_PROMOTER_FEATURES if (a.legacy_features and a.legacy_init) else (),
                   dist_tss=a.legacy_features,
                   kmer_k=a.kmer_k, snapshot=snapshot, supervision=a.supervision,
                   kmer_transitions=kt, min_len=min_len, max_tokens=a.max_tokens)
    if a.l2_kmer is not None:
        cfg.l2["vk"] = a.l2_kmer
        cfg.l2["vkt"] = a.l2_kmer
    m = TxHMM(cfg, seed=a.seed)
    lb = m.lays_fit.get("body", m.lays_fit["full"])
    print(f"[fit] initiation={cfg.initiation}; {m.lay.S} states, {m.lay.E} transitions "
          f"({lb.S} states / {len(lb.ed)} position-dependent transitions along reads); "
          f"{sum(v.size for v in m.p.values()):,} parameters", flush=True)
    bad = set(cfg.freeze) - set(m.p)
    if bad:
        raise SystemExit(f"--freeze: unknown parameter blocks {sorted(bad)}; known: {sorted(m.p)}")
    if a.init_from:
        m.init_from_model(a.init_from, verbose=True)
    rtr = ReadSet.from_dataset(train)
    rte = ReadSet.from_dataset(test) if test is not None else None
    t0 = time.perf_counter()
    m.fit(rtr, n_iter=a.iters, batch_size=a.batch, lr=a.lr, val=rte, log_every=a.log_every,
          n_workers=a.workers, tasks_per_worker=a.tasks_per_worker, val_max_reads=a.val_reads,
          patience=a.patience, patience_metric=a.patience_metric, tol=a.tol,
          init_from_data=not a.init_from)
    print(f"[time] fate model: {time.perf_counter() - t0:.1f} s", flush=True)
    m.save(a.out + ".tx.npz")
    if init_proc is not None:
        init_proc.join()
    print(json.dumps(m.summary(), indent=1))
    print(f"[fit] saved {a.out}.tx.npz" + (f" and {a.out}.init.npz" if a.legacy_init else ""))


def _region(s):
    chrom, rest = s.rsplit(":", 1)
    start, end = rest.replace(",", "").split("-")
    return chrom, int(start), int(end)


def cmd_predict(a):
    from .model import TxHMM, InitModel
    from .predict import predict_genomic, write_bedgraphs
    tx = TxHMM.load(a.tx)
    im = None
    if not tx.cfg.chain:
        if not a.init:
            raise SystemExit("this model uses legacy initiation: pass --init PREFIX.init.npz")
        im = InitModel.load(a.init)
    elif a.init:
        print("[predict] --init is only used by legacy-initiation models; ignored")
    chrom, start, end = _region(a.region)
    starts = [int(x) for x in a.start.split(",")] if a.start else None
    if a.introns and a.introns_from:
        raise SystemExit("pass --introns or --introns-from, not both")
    introns = None
    if a.introns:
        introns = _read_introns(a.introns, chrom, start, end)
    elif a.introns_from:
        from .structures import region_introns
        introns = region_introns(a.introns_from, chrom, start, end)
    known = getattr(tx.cfg, "known_introns", False)
    if introns is not None and not known:
        print("[predict] --introns / --introns-from are only used by --known-introns models; ignored")
        introns = None
    if known and introns is None:
        print("[predict] WARNING: --known-introns model without --introns / --introns-from: "
              "the region is treated as having no intron at all")
    if starts and a.strand == "both":
        raise SystemExit("--start needs a single --strand (+ or -)")
    fixed = a.tss or a.tss_from
    if fixed:
        if not tx.cfg.chain:
            raise SystemExit("--tss / --tss-from need a chain-initiation model")
        if a.tss and a.tss_from:
            raise SystemExit("pass --tss or --tss-from, not both")
        if starts:
            raise SystemExit("--start and --tss / --tss-from are exclusive (Pol II is placed "
                             f"{tx.cfg.upstream} nt before each TSS)")
        if a.strand == "both":
            raise SystemExit("--tss / --tss-from need a single --strand (+ or -)")
    strands = ["+", "-"] if a.strand == "both" else [a.strand]
    summary = {}
    for st in strands:
        kw = dict(starts=starts) if tx.cfg.chain else dict(n_tss=a.n_tss)
        if a.tss:
            kw = dict(tss=[int(x) for x in a.tss.replace(",", " ").split()])
        elif a.tss_from:
            from .structures import region_caps
            cpos, cn = region_caps(a.tss_from, chrom, start, end, st)
            if not len(cpos):
                raise SystemExit(f"no read cap of {a.tss_from} inside {a.region} on strand {st}")
            if a.tss_caps == "main":
                kw = dict(tss=[int(cpos[np.argmax(cn)])])
            else:
                kw = dict(tss=[int(x) for x in cpos], tss_weights=cn.astype(float))
            print(f"[predict] strand {st}: {cn.sum():,} read caps at {len(cpos):,} positions; main cap "
                  f"{int(cpos[np.argmax(cn)]):,} ({cn.max():,} reads); using "
                  f"{'the main cap' if a.tss_caps == 'main' else 'all caps, weighted by reads'}")
        if introns is not None:
            kw["introns"] = [(s_, e_) for s_, e_, x in introns if x in (st, ".")]
            print(f"[predict] strand {st}: {len(kw['introns'])} known introns")
        res = predict_genomic(tx, im, a.fasta, chrom, start, end, strand=st, flank=a.flank, **kw)
        paths = write_bedgraphs(res, a.out)
        summary[st] = dict(tss=res["tss"], tss_mass=res["tss_mass"], files=paths)
        if tx.cfg.chain and res.get("fixed_tss"):
            ft = res["fixed_tss"]
            summary[st].update(fixed_tss=ft, terminated_in_region=res["terminated_in_region"])
            cm = np.sum([np.array(t["class_probs"]) * t["weight"] for t in ft], 0)
            pf = sum(t["p_fire_here"] * t["weight"] for t in ft)
            where = f"{ft[0]['genomic']:,}" if len(ft) == 1 else f"{len(ft)} TSSs"
            print(f"[predict] strand {st}: fires at {where} (conditioned; Pol II {tx.cfg.upstream} nt upstream); "
                  f"class mix {' / '.join(f'{x:.2f}' for x in cm)}; the model's own P(first firing there) "
                  f"{pf:.3g}; {res['terminated_in_region']:.2f} terminated inside the region; "
                  f"wrote {len(paths)} bedGraphs")
        elif tx.cfg.chain:
            q = res.get("firing_quantiles", {})
            summary[st].update(pol2_starts=res["starts"], fired_in_region=res["fired_in_region"],
                               never_fired=res["never_fired"], firing_quantiles=q,
                               terminated_in_region=res["terminated_in_region"])
            fq = (f"; fires at {q[0.5]:,} (10-90%: {min(q[0.1], q[0.9]):,}-{max(q[0.1], q[0.9]):,})"
                  if q else "")
            print(f"[predict] strand {st}: Pol II placed at {', '.join(f'{x:,}' for x in res['starts'])}; "
                  f"fired inside the region {res['fired_in_region']:.2f}{fq}; "
                  f"{res['terminated_in_region']:.2f} terminated inside the region; wrote {len(paths)} bedGraphs")
        else:
            print(f"[predict] strand {st}: {len(res['tss'])} TSSs "
                  f"(carry {res['tss_mass']:.2f} of the initiation propensity); wrote {len(paths)} bedGraphs")
    with open(a.out + ".tss.json", "w") as fh:
        json.dump(summary, fh, indent=1)


def _read_introns(path, chrom, start, end):
    """BED (chrom, start, end[, name, score, strand]) -> [(start, end, strand)] inside the region."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            if f[0] == chrom and int(f[1]) >= start and int(f[2]) <= end:
                out.append((int(f[1]), int(f[2]), f[5] if len(f) > 5 else "."))
    return out


def cmd_structures(a):
    from .structures import build
    build(a.source, a.data, a.out, a.min_junction_reads, a.min_intron)


def _print_table(tab, order, cols, fmt):
    print("  " + " ".join(f"{c:>{w}s}" for c, w in zip(cols, fmt)))
    for i in order:
        vals = []
        for c, w in zip(cols, fmt):
            v = tab[c][i]
            vals.append(f"{v:>{w}s}" if isinstance(v, str) else f"{v:>{w}.3g}")
        print("  " + " ".join(vals))


def cmd_inspect(a):
    from .model import TxHMM, InitModel
    from .dataprep import TxDataset
    from .interpret import (kmer_table, model_pwms, consensus, write_tsv, transition_report)
    tx = TxHMM.load(a.tx)
    c = tx.cfg
    im = InitModel.load(a.init) if a.init else None
    ds = TxDataset.load(a.data) if a.data else None
    out = a.out
    summary = {"model": json.loads(c.to_json()), "parameters": tx.summary()}

    # ---- termination, random route
    if c.kmer_k:
        t = kmer_table(tx, ds)
        for col in [k for k in t if k.startswith("effect_")]:
            for direction, order in (("raise", np.argsort(-t[col])), ("lower", np.argsort(t[col]))):
                print(f"\n[termination, random route, {col[7:]}] k-mers (last {c.kmer_k} nt transcribed) "
                      f"that {direction} the hazard most:")
                cols = ["kmer", col] + [k for k in t if k.startswith("hazard_per_nt_")] + \
                       (["n_read_ends", "n_positions", "end_enrichment"] if ds is not None else [])
                _print_table(t, order[:a.top], cols, [8] + [12] * (len(cols) - 1))
        if out:
            write_tsv(out + ".termination_kmers.tsv", t)

    # ---- transitions
    rep = transition_report(tx, ds, n_reads=a.support_reads)
    print("\n[baseline] transition probabilities without k-mer effects:")
    for r in rep["baseline"]:
        pr = "" if not np.isfinite(r["prob"]) else f"{r['prob']:9.3g}"
        print(f"  {r['kind']:20s} {r['track']:8s} {pr:>9s}  {r['meaning']}")
    if "promoter_chain" in rep:
        pc = rep["promoter_chain"]
        print("\n[pre-initiation chain] per-nt transition probabilities of each promoter sub-state "
              "(no k-mer effect):")
        _print_table(pc, range(len(pc["state"])), ["state", "stay", "advance", "exit", "initiate"], [8, 9, 9, 9, 9])
        if out:
            write_tsv(out + ".promoter_chain.tsv", pc)
    for key in ("posterior_P(track|class)", "posterior_class_share", "terminations_at_read_ends"):
        if key in rep:
            print(f"\n[posterior, {rep['support']['n_reads']} reads] {key}: {rep[key]}")
            summary[key] = rep[key]
    for T, tab in rep.get("cleavage_by_state", {}).items():
        print(f"\n[cleavage entry by splicing state, track {T}] offset = learned log-odds vs unspliced; "
              f"P(enter | k-mer) per nt, away from splice sites")
        cols = list(tab)
        _print_table(tab, range(len(tab["state"])), cols, [13] + [max(12, len(c_) + 1) for c_ in cols[1:]])
        if out:
            write_tsv(f"{out}.cleavage_by_state_{T}.tsv", tab)
        summary.setdefault("cleavage_by_state", {})[T] = {k: (v if isinstance(v, list) else v.tolist())
                                                          for k, v in tab.items()}
    top_rows = []
    for name, tab in rep["tables"].items():
        g = rep["groups"][name]
        cols = ["kmer", "effect", "prob_base", "prob_with_kmer"] + \
               (["expected_transitions", "exposure", "observed_rate"] if "exposure" in tab else [])
        for direction, order in (("raise", np.argsort(-tab["effect"])), ("lower", np.argsort(tab["effect"]))):
            if direction == "lower" and not a.show_lower:
                continue
            print(f"\n[{name}] k-mers that {direction} it most ({g['where']}):")
            _print_table(tab, order[:a.top], cols, [8] + [12] * (len(cols) - 1))
            for rank, i in enumerate(order[:a.top]):
                top_rows.append([name, direction, rank + 1] + [tab[k][i] for k in cols])
        if out:
            write_tsv(f"{out}.transition_{name}.tsv", tab)
    if out and top_rows:
        cols = ["transition", "direction", "rank", "kmer", "effect", "prob_base", "prob_with_kmer",
                "expected_transitions", "exposure", "observed_rate"]
        with open(out + ".top_kmers.tsv", "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in top_rows:
                fh.write("\t".join(x if isinstance(x, str) else (str(x) if isinstance(x, int) else f"{x:.4g}")
                                   for x in r) + "\n")

    # ---- implied distributions and posterior profiles
    for name, (pmf, s) in rep["distributions"].items():
        summary[name] = s
        if out:
            write_tsv(f"{out}.{name}.tsv", {"nt": np.arange(1, len(pmf) + 1), "probability": pmf})
    if rep["support"] and out:
        W = rep["support"]["W"]
        for name, h in rep["support"]["prof"].items():
            pos = np.arange(len(h)) - (0 if name.startswith("cleave_entry") else W)
            label = "nt_upstream_of_read_end" if name.startswith("cleave_entry") else "position_relative_to_cap"
            keep = h > 0
            write_tsv(f"{out}.profile_{name}.tsv", {label: pos[keep], "expected_count": h[keep]})
    for name, h in (rep["support"]["prof"].items() if rep["support"] else []):
        if name.startswith("cleave_entry") and h.sum() > 0:
            d = np.arange(len(h)); cdf = np.cumsum(h) / h.sum()
            print(f"\n[posterior] {name}: cleavage path entered a median {int(d[np.searchsorted(cdf, .5)])} nt "
                  f"upstream of the read 3' end (10-90%: {int(d[np.searchsorted(cdf, .1)])}-"
                  f"{int(d[np.searchsorted(cdf, .9)])}), {h.sum():.1f} reads")
        if name.startswith("penter_class") and h.sum() > 0:
            W = rep["support"]["W"]
            pos = np.arange(len(h)) - W
            cdf = np.cumsum(h) / h.sum()
            q = lambda x: int(pos[np.searchsorted(cdf, x)])
            print(f"\n[posterior] {name}: promoter class entered a median {q(.5):+d} nt from the cap "
                  f"(10-90%: {q(.1):+d}..{q(.9):+d}; the walk starts at {-W:+d}), {h.sum():.1f} entries")

    for name, tab in model_pwms(tx, im).items():
        print(f"\n[{name} PWM] {tab['anchor']}; consensus (upper = strong, lower = weak, n = none):")
        offs = tab["offset"]
        print(f"  offsets {offs[0]:+d}..{offs[-1]:+d}: {consensus(tab)}")
        if out:
            write_tsv(f"{out}.{name}_pwm.tsv", tab)
    if out:
        with open(out + ".summary.json", "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        print(f"\n[inspect] tables written with prefix {out}")


def cmd_simulate(a):
    from .simulate import simulate
    simulate(a.outdir, n_loci=a.loci, reads_per_locus=a.reads, seed=a.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="txhmm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="BAM/BED12 + FASTA -> dataset.npz")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bam"); src.add_argument("--bed12")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-mapq", type=int, default=0)
    p.add_argument("--min-intron", type=int, default=20)
    p.add_argument("--strand-mode", default="read", help="read | reverse | ts | tag:XS")
    p.add_argument("--cap-mode", default="all", help="all | tag:NAME | name:REGEX")
    p.add_argument("--end-mode", default="none",
                   help="none | tag:NAME:THRESH | softclip:MINLEN:MINFRAC | name:REGEX")
    p.add_argument("--end-types", default="pA-:mixture,pA+:terminated",
                   help="class:type pairs; type in terminated|censored|mixture|survived|drop")
    p.add_argument("--weight-tag")
    p.add_argument("--flank", type=int, default=1000)
    p.add_argument("--snap", type=int, default=5)
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=50000)
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("fit", help="fit the model")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--supervision", default="full", choices=["full", "status", "none"])
    p.add_argument("--tracks", default="A,P",
                   help="fate tracks, comma separated: 'A,P' (two populations of polymerases, chosen in "
                        "the early phase) or 'P' (a single track: every polymerase follows the same rules)")
    p.add_argument("--switching", action="store_true")
    p.add_argument("--no-splicing", action="store_true")
    p.add_argument("--n-wait", type=int, default=1)
    p.add_argument("--min-intron", type=int, default=30)
    g = p.add_argument_group("initiation (pre-initiation chain)")
    g.add_argument("--upstream", type=int, default=100,
                   help="nt of DNA the pre-initiation chain walks before the cap")
    g.add_argument("--start-spread", type=int, default=0,
                   help="> 1: Pol II starts uniformly over the first N positions of the walk, i.e. "
                        "between cap - upstream and cap - upstream + N - 1 (0 = exactly at cap - upstream)")
    g.add_argument("--init-window-down", type=int, default=100,
                   help="P(initiation at the cap) is relative to [cap - upstream, cap + this] (--init-score window)")
    g.add_argument("--init-score", default="window", choices=["window", "absolute"],
                   help="initiation term of each read: 'window' = log P(first firing at the cap) - log P(first "
                        "firing in [cap - upstream, cap + init-window-down]) (old behaviour); 'absolute' = "
                        "log P(first firing exactly at the cap), no normaliser")
    g.add_argument("--promoter-classes", type=int, default=2)
    g.add_argument("--promoter-substates", type=int, default=10)
    g.add_argument("--early-substates", type=int, default=3,
                   help="sub-states of the early phase (after initiation, before commitment)")
    g.add_argument("--upstream-lookahead", type=int, default=3,
                   help="pre-initiation transitions at u read the k-mer ending at u + this")
    g.add_argument("--class-track-lean", type=float, default=0.5,
                   help="starting lean of the promoter classes towards the tracks (two tracks): "
                        "0.5 = class 0 towards P, last class towards A; 0 = none; negative = reversed")
    g.add_argument("--legacy-init", action="store_true",
                   help="old initiation: INIT state + separate initiation model (PREFIX.init.npz)")
    p.add_argument("--cleave-substates", type=int, default=8,
                   help="sub-states of the k-mer-entered cleavage path (0 = none)")
    p.add_argument("--init-from", default=None,
                   help="start from this fitted model's parameters (.tx.npz; same structure)")
    p.add_argument("--freeze", default="",
                   help="comma list of parameter blocks kept at their starting values, e.g. "
                        "u_don,u_acc,b_don,b_acc,b_splice,b_abandon (use with --init-from)")
    p.add_argument("--known-introns", action="store_true",
                   help="donors/acceptors only at the dataset's known intron positions (txhmm structures); "
                        "implies --all-introns")
    p.add_argument("--all-introns", action="store_true",
                   help="track every intron (later introns: own termination offsets, shared splicing parameters)")
    p.add_argument("--fix-intron-offsets", nargs="?", const="all", default=False, choices=["all", "first"],
                   help="keep the 'in intron' termination offsets (random route and cleavage entry) at 0, i.e. equal "
                        "to exon 1: 'all' (default when given without a value) = first and later introns, "
                        "'first' = first intron only")
    p.add_argument("--legacy-features", action="store_true",
                   help="add the hand-made features of the first version: GC windows, T richness, "
                        "PAS window and distance to the TSS in the hazard (and promoter GC / TATA "
                        "with --legacy-init)")
    p.add_argument("--kmer-k", type=int, default=5)
    p.add_argument("--kmer-transitions", default="",
                   help="comma list of splicing transitions that also get k-mer effects: donor,acceptor,"
                        "abandon,splice,switch (commit with --legacy-init), or 'all'; default none")
    p.add_argument("--l2-kmer", type=float, default=None,
                   help="L2 prior (precision) on every k-mer table entry (default 10); lower it "
                        "(e.g. 1) for small datasets with strong k-mer effects")
    p.add_argument("--snapshot", action="store_true",
                   help="learn a snapshot (harvest) hazard; needed for 'mixture' ends")
    p.add_argument("--min-len", type=int, default=0, help="default: the value used in prep")
    p.add_argument("--iters", type=int, default=1000,
                   help="Adam steps; each uses workers x tasks-per-worker x batch reads")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--tasks-per-worker", type=int, default=2,
                   help="batches per worker per step (load balancing); reads per step = "
                        "workers x tasks-per-worker x batch")
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=150000,
                   help="max reads x nt per batch; memory per worker grows with it "
                        "(~ 2-3 kB per token with the default model)")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=0,
                   help="stop when the validation line has not improved for this many log points "
                        "(0 = run all --iters); the best parameters are kept")
    p.add_argument("--patience-metric", choices=("per_nt", "per_read"), default="per_nt",
                   help="validation value used by --patience: per_nt (old behaviour; its absolute tolerance "
                        "becomes too strict with long reads) or per_read")
    p.add_argument("--tol", type=float, default=None,
                   help="minimum improvement counted by --patience (default 1e-6 per nt / 1e-3 nats per read)")
    p.add_argument("--val-reads", type=int, default=5000,
                   help="held-out reads used for the validation line printed during fitting")
    p.add_argument("--init-pad", type=int, default=150, help="legacy initiation model only")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("predict", help="profiles for a genomic region from sequence only")
    p.add_argument("--tx", required=True)
    p.add_argument("--init", help="legacy-initiation models only")
    p.add_argument("--fasta", required=True)
    p.add_argument("--region", required=True, help="chr:start-end (0-based, end exclusive)")
    p.add_argument("--strand", default="both", choices=["+", "-", "both"])
    p.add_argument("--start", help="where Pol II is placed (genomic, 0-based; several separated by commas), "
                                   "default: the region's 5' end on the strand. The model was fitted "
                                   "with Pol II 100 nt upstream of each cap (--upstream)")
    p.add_argument("--tss", help="TSS(s) where Pol II fires (genomic, 0-based first transcribed nt; several "
                                 "separated by commas, equal shares). Firing is conditioned on the TSS as in "
                                 "training: Pol II is placed --upstream nt before it and only the walks firing "
                                 "exactly there are kept; the walk only sets the promoter class mix")
    p.add_argument("--tss-from", help="dataset .npz: take the TSS(s) from its read caps inside the region "
                                      "(see --tss-caps), instead of --tss")
    p.add_argument("--tss-caps", default="main", choices=["main", "all"],
                   help="with --tss-from: the most frequent cap only, or every cap weighted by its reads")
    p.add_argument("--n-tss", type=int, default=10, help="legacy-initiation models only")
    p.add_argument("--introns", help="BED of known introns (genomic, 0-based half-open; strand in column 6, "
                                     "'.' = both) for --known-introns models")
    p.add_argument("--introns-from", help="dataset .npz with known introns (txhmm structures): use its "
                                          "introns lying inside the region, instead of --introns")
    p.add_argument("--flank", type=int, default=1000, help="sequence context fetched around the region")
    p.add_argument("--out", required=True, help="output prefix")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("structures", help="known intron structures from read junctions, added to a dataset")
    p.add_argument("--source", required=True, help="dataset whose junctions define the structures")
    p.add_argument("--data", required=True, help="dataset that receives them (its loci)")
    p.add_argument("--out", required=True, help="new dataset file (.npz)")
    p.add_argument("--min-junction-reads", type=int, default=2)
    p.add_argument("--min-intron", type=int, default=30)
    p.set_defaults(func=cmd_structures)

    p = sub.add_parser("inspect", help="top k-mers of every transition, termination, dwell "
                                       "distributions and splice-site PWMs of a fitted model")
    p.add_argument("--tx", required=True)
    p.add_argument("--init", help="legacy initiation model (adds its PWM)")
    p.add_argument("--data", help="dataset.npz: adds k-mer occurrences and the posterior support "
                                  "(expected transitions / exposure per k-mer, position profiles)")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--show-lower", action="store_true",
                   help="also print the k-mers that lower each transition most (always in the TSVs)")
    p.add_argument("--support-reads", type=int, default=1000,
                   help="reads used for the posterior support (needs --data)")
    p.add_argument("--out", help="prefix for the TSV/JSON tables")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("simulate", help="synthetic data with known truth")
    p.add_argument("--outdir", required=True)
    p.add_argument("--loci", type=int, default=200)
    p.add_argument("--reads", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main(sys.argv[1:])
