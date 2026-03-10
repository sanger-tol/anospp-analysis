import subprocess
from pathlib import Path
import pandas as pd

def run_command(cmd):
    """
    Run a command and return (stdout, stderr).
    Raises RuntimeError on failure.
    """
    try:
        result = subprocess.run(
            list(cmd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with exit code {e.returncode}\n"
            f"CMD: {' '.join(e.cmd)}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        ) from e

    return result.stdout, result.stderr

def run_vsearch_sintax(
    fasta_in,
    db,
    tsv_out,
    *,
    random=True,
    threads=None,
    vsearch_bin="vsearch",
):
    fasta_in = Path(fasta_in)
    db = Path(db)
    tsv_out = Path(tsv_out)

    cmd = [
        vsearch_bin,
        "--sintax", str(fasta_in),
        "--db", str(db),
        "--tabbedout", str(tsv_out),
    ]

    if random:
        cmd.append("--sintax_random")

    if threads is not None:
        cmd.extend(["--threads", str(threads)])

    return run_command(cmd)

def run_vsearch_cluster_fast(
    fasta_in,
    identity=0.97,
    centroids=None,
    clusters=None,
    threads=None,
    vsearch_bin="vsearch",
):
    """
    Run vsearch --cluster_fast.

    At least one of `centroids` or `clusters` must be specified.
    """

    if centroids is None and clusters is None:
        raise ValueError(
            "At least one of `centroids` or `clusters` must be specified"
        )

    cmd = [
        vsearch_bin,
        "--cluster_fast", str(fasta_in),
        "--id", str(identity),
    ]

    if centroids is not None:
        cmd.extend(["--centroids", str(Path(centroids))])

    if clusters is not None:
        cmd.extend(["--clusters", str(Path(clusters))])

    if threads is not None:
        cmd.extend(["--threads", str(threads)])

    return run_command(cmd)

def parse_sintax(sintax_tsv, ranks):

    df = pd.read_csv(sintax_tsv, sep="\t", names=["seqid", "taxonomy", "strand"])
    
    tax = (
        df[["seqid", "taxonomy"]]
        .assign(taxonomy=lambda x: x["taxonomy"].str.split(","))
        .explode("taxonomy")
        .assign(taxonomy=lambda x: x["taxonomy"].str.strip())
        .assign(
            rank=lambda x: x["taxonomy"].str.extract(r"^(\w):"),
            name=lambda x: x["taxonomy"].str.extract(r":(.+)\(")[0],
            bootstrap=lambda x: x["taxonomy"].str.extract(r"\(([\d.]+)\)")[0].astype(float),
        )
    )
    
    tax["rank"] = tax["rank"].map(ranks)
    
    wide = (
        tax
        .pivot(index="seqid", columns="rank", values="name")
        .join(
            tax.pivot(index="seqid", columns="rank", values="bootstrap")
            .add_suffix("_bootstrap")
        )
    )
    
    df = df.merge(wide, left_on="seqid", right_index=True)

    out_cols = [
        'seqid',
        'strand'
    ] 
    for r in ranks.values():
        out_cols.extend([r, r + '_bootstrap'])
    
    for col in out_cols:
        if col not in df.columns:
            df[col] = ''

    return df[out_cols].copy()