"""Tests for the cross-dataset spatial aggregation (derived datasets)."""
import geopandas as gpd
from shapely.geometry import box, Point

from catalyst._internal.server.derive.spatial_aggregate import spatial_aggregate


def test_dominant_is_area_weighted_mode():
    # Park A is covered 3:1 by forest vs grass -> dominant = forest.
    # Park B is covered only by desert -> dominant = desert.
    parks = gpd.GeoDataFrame(
        {"PARK_NAME": ["A", "B"]},
        geometry=[box(0, 0, 2, 2), box(10, 10, 12, 12)],
        crs=None,
    )
    veg = gpd.GeoDataFrame(
        {"VEG_TYPE": ["forest", "grass", "desert"]},
        geometry=[box(0, 0, 1.5, 2), box(1.5, 0, 2, 2), box(10, 10, 12, 11)],
        crs=None,
    )

    out = spatial_aggregate(
        parks, veg,
        predicate="intersects", aggregate="dominant",
        value_attribute="VEG_TYPE", weight="area",
        output_attribute="dominant_veg",
    )

    assert "dominant_veg" in out.columns
    assert len(out) == 2  # exactly one row per target feature
    vals = dict(zip(out["PARK_NAME"], out["dominant_veg"]))
    assert vals["A"] == "forest"
    assert vals["B"] == "desert"


def test_count_points_within_each_polygon():
    parks = gpd.GeoDataFrame(
        {"PARK_NAME": ["A", "B"]},
        geometry=[box(0, 0, 2, 2), box(10, 10, 12, 12)],
        crs=None,
    )
    pts = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[Point(0.5, 0.5), Point(1.5, 1.5), Point(10.5, 10.5)],
        crs=None,
    )

    out = spatial_aggregate(
        parks, pts,
        predicate="intersects", aggregate="count",
        value_attribute=None, weight=None,
        output_attribute="n_points",
    )

    vals = dict(zip(out["PARK_NAME"], out["n_points"]))
    assert vals["A"] == 2
    assert vals["B"] == 1


def test_target_with_no_match_gets_null_not_dropped():
    parks = gpd.GeoDataFrame(
        {"PARK_NAME": ["A", "Lonely"]},
        geometry=[box(0, 0, 2, 2), box(100, 100, 102, 102)],
        crs=None,
    )
    veg = gpd.GeoDataFrame(
        {"VEG_TYPE": ["forest"]},
        geometry=[box(0, 0, 2, 2)],
        crs=None,
    )

    out = spatial_aggregate(
        parks, veg,
        predicate="intersects", aggregate="dominant",
        value_attribute="VEG_TYPE", weight="area",
        output_attribute="dominant_veg",
    )

    assert len(out) == 2  # the unmatched target is preserved, not dropped
    vals = dict(zip(out["PARK_NAME"], out["dominant_veg"]))
    assert vals["A"] == "forest"
    assert vals["Lonely"] is None or vals["Lonely"] != vals["Lonely"]  # None or NaN
