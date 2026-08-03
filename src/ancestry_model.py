"""
Toy model: predict 1000 Genomes population from haploblock cluster assignment.

This is the local/testable version of the model — it reads from a local
directory of `individual_hashes_START-END.tsv` files instead of S3, so the
logic can be validated before being dropped into the AWS get-to-know-a-dataset
notebook (where the local glob is swapped for a boto3 list_objects_v2 loop).
"""

import glob
import os

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_individual_hashes(local_dir, max_regions=None):
    """Read individual_hashes_*.tsv files into one long-format table.

    Each row in the source files is one haplotype (hap0 or hap1) of one
    sample. This collapses the two haplotypes per sample/region into a
    single unphased "genotype" string (sorted so hap order doesn't matter),
    since a sample's diploid genotype at a haploblock -- not a lone
    haplotype -- is the natural unit for a per-individual feature.

    HASH values are kept as a sorted list here (not joined into a string)
    because polars' `str.join` aggregation isn't available in all polars
    versions -- the list gets joined into a string in pandas instead, in
    build_feature_matrix, which is stable across versions.
    """
    paths = sorted(glob.glob(os.path.join(local_dir, "individual_hashes_*.tsv")))
    if not paths:
        raise FileNotFoundError(
            f"No individual_hashes_*.tsv files found under {local_dir!r}. "
            "Check the directory path and that files are named exactly "
            "'individual_hashes_START-END.tsv'."
        )
    if max_regions is not None:
        paths = paths[:max_regions]

    frames = []
    for path in paths:
        region = os.path.basename(path).replace("individual_hashes_", "").replace(".tsv", "")
        df = pl.read_csv(path, separator="\t", schema_overrides={"HASH": pl.Utf8})
        df = df.with_columns(
            [
                pl.col("INDIVIDUAL").str.extract(r"^([^_]+)", 1).alias("sample_id"),
                pl.lit(region).alias("region"),
            ]
        )
        genotype = (
            df.group_by(["sample_id", "region"])
            .agg(pl.col("HASH").sort())
        )
        frames.append(genotype)

    return pl.concat(frames)


def load_population_panel(igsr_dir):
    """Read IGSR per-population export files (e.g. igsr-chb.tsv, igsr-gbr.tsv,
    igsr-pur.tsv) and build a sample_id -> population lookup.

    Real IGSR sample-data-portal exports already carry their own
    'Population code' column, so files are just concatenated -- population
    is not inferred from the filename.
    """
    paths = sorted(glob.glob(os.path.join(igsr_dir, "igsr-*.tsv")))
    frames = [pd.read_csv(p, sep="\t") for p in paths]
    panel = pd.concat(frames, ignore_index=True)
    return panel.set_index("Sample name")["Population code"].to_dict()


def build_feature_matrix(long_df, igsr_dir):
    """Pivot to sample x region, attach population labels, sparse-encode.

    Uses a sparse one-hot representation instead of pandas.get_dummies:
    at full-chromosome scale (thousands of regions), a dense one-hot matrix
    becomes very large very fast, even though each row only has one non-zero
    entry per region. Sparse encoding keeps memory proportional to the
    actual number of (sample, region) observations rather than
    samples x total_categories_across_all_regions.
    """
    sample_to_pop = load_population_panel(igsr_dir)

    wide_df = long_df.to_pandas().pivot(index="sample_id", columns="region", values="HASH")
    join_list = lambda h: "|".join(h) if isinstance(h, (list, tuple, np.ndarray)) else h
    wide_df = wide_df.apply(lambda col: col.map(join_list))
    wide_df["population"] = wide_df.index.map(sample_to_pop)
    wide_df = wide_df.dropna(subset=["population"])

    region_cols = [c for c in wide_df.columns if c != "population"]
    blocks = []
    feature_regions = []  # parallel list: region name for each output column
    n_rows = len(wide_df)

    for region in region_cols:
        codes, uniques = pd.factorize(wide_df[region])
        n_categories = len(uniques)
        rows = np.arange(n_rows)
        data = np.ones(n_rows)
        block = sparse.csr_matrix((data, (rows, codes)), shape=(n_rows, n_categories))
        blocks.append(block)
        feature_regions.extend([region] * n_categories)

    X = sparse.hstack(blocks, format="csr")
    y = wide_df["population"]
    return X, y, feature_regions


def train_and_evaluate(X, y, feature_regions):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    # RandomForestClassifier accepts sparse input directly (converted
    # internally to csc); no dense conversion needed even at large scale.
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(X_train, y_train)
    report = classification_report(y_test, clf.predict(X_test))

    importances = pd.Series(clf.feature_importances_)
    region_importance = (
        importances.groupby(feature_regions).sum().sort_values(ascending=False)
    )

    return clf, report, region_importance


if __name__ == "__main__":
    # Matches the real local layout:
    # haploblocks-ancestry-model/test_data/hashes/individual_hashes_*.tsv
    # haploblocks-ancestry-model/test_data/igsr/igsr-{chb,gbr,pur}.tsv
    #
    # Paths are resolved relative to this script's location (not the cwd),
    # so `python3 ancestry_model.py` works the same whether launched from
    # src/ or the repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(repo_root, "test_data", "hashes")
    igsr_dir = os.path.join(repo_root, "test_data", "igsr")

    total_available = len(glob.glob(os.path.join(local_dir, "individual_hashes_*.tsv")))
    # Bump this up incrementally (e.g. 200, 500, ...) or set to None for the
    # full chromosome. Sparse encoding means memory scales with actual
    # (sample, region) observations, not samples x total categories.
    MAX_REGIONS = 2287
    print(f"Using {min(MAX_REGIONS, total_available) if MAX_REGIONS else total_available} "
          f"of {total_available} available region files")

    long_df = load_individual_hashes(local_dir, max_regions=MAX_REGIONS)
    print(f"Loaded {long_df['region'].n_unique()} regions, {long_df.height} rows")

    X, y, feature_regions = build_feature_matrix(long_df, igsr_dir)
    print(f"Feature matrix: {X.shape} (sparse, {X.nnz} non-zero entries), "
          f"label counts:\n{y.value_counts()}")

    clf, report, region_importance = train_and_evaluate(X, y, feature_regions)
    print("\nClassification report:\n", report)
    print("\nTop regions by aggregated importance:\n", region_importance.head())
