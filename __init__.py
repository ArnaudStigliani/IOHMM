"""txhmm: sequence-driven model of initiation, splicing and termination (NumPy, CPU)."""
import os as _os

# One BLAS/OpenMP thread per process: parallel training uses worker processes, and the
# matrices in the forward-backward are tiny. Only effective if numpy is not imported yet;
# otherwise export OMP_NUM_THREADS=1 (and OPENBLAS/MKL) before starting Python.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
from .genome import FastaFile, iter_bam, iter_bed12, write_bam, write_bed12, write_fasta, revcomp
from .sequence import SequenceStore, FeatureDef
from .dataprep import PrepConfig, TxDataset, dataset_from_bam, dataset_from_bed12
from .model import TxConfig, TxHMM, InitModel, ReadSet, check_gradients

__all__ = ["FastaFile", "iter_bam", "iter_bed12", "write_bam", "write_bed12", "write_fasta",
           "revcomp", "SequenceStore", "FeatureDef", "PrepConfig", "TxDataset",
           "dataset_from_bam", "dataset_from_bed12", "TxConfig", "TxHMM", "InitModel",
           "ReadSet", "check_gradients"]
