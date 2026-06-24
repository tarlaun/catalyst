"""Integration test: derive a dataset by spatial-aggregating two built datasets."""
import os

import geopandas as gpd
import pytest

from catalyst._internal.server.derive.spatial_aggregate import (
    derive_dataset,
    load_dataset_gdf,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets")
PARKS = os.path.join(DATA_ROOT, "riverside_parks")
VEG = os.path.join(DATA_ROOT, "riverside_vegetation_types")

requires_data = pytest.mark.skipif(
    not (os.path.isdir(PARKS) and os.path.isdir(VEG)),
    reason="riverside_parks / riverside_vegetation_types datasets not built",
)


@requires_data
def test_load_dataset_gdf_returns_geometries():
    gdf = load_dataset_gdf(PARKS)
    assert len(gdf) > 0
    assert "geometry" in gdf.columns
    assert gdf.geometry.notna().any()
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326


@requires_data
def test_derive_parks_dominant_vegetation(tmp_path):
    out_path = str(tmp_path / "parks_dom_veg.parquet")
    result_path = derive_dataset(
        data_root=DATA_ROOT,
        target_dataset="riverside_parks",
        source_dataset="riverside_vegetation_types",
        predicate="intersects",
        aggregate="dominant",
        value_attribute="FIRST_CATE",
        weight="area",
        output_attribute="dominant_vegetation",
        out_path=out_path,
    )
    gdf = gpd.read_parquet(result_path)
    n_parks = len(load_dataset_gdf(PARKS))

    assert "dominant_vegetation" in gdf.columns
    assert len(gdf) == n_parks                       # one row per park, none dropped
    assert gdf["dominant_vegetation"].notna().sum() > 0  # some parks got a veg type
    # values come from the source vegetation categories
    veg = load_dataset_gdf(VEG)
    assert set(gdf["dominant_vegetation"].dropna()).issubset(set(veg["FIRST_CATE"].dropna()))
