from datetime import date, timedelta
import csv

import pytest
import torch

from complexity_ensemble.app import BasinMapComponent, DEFAULT_BASINS, _config_from_ui_values
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.regional import (
    BasinCatalogRecord, BasinScopeConfig, CAMELSCHCatalog, CancellationToken,
    DateRange, DateSplitConfig, ExtremeEventConfig, HyperparameterConfig,
    ModelArchitectureConfig, RegionalExperimentConfig, TrainingStrategyConfig,
    make_regional_hydrology_data, run_regional_experiment,
)


def _series(basin: str, offset: float = 0.0) -> HydrologySeries:
    first = date(2001, 1, 1)
    dates = tuple(first + timedelta(days=index) for index in range(65))
    forcing = torch.stack((
        torch.tensor([2.0 if index % 5 == 0 else 0.2 for index in range(65)]),
        torch.linspace(2, 10, 65), torch.full((65,), 0.5),
    ), dim=1)
    return HydrologySeries(dates, torch.linspace(1 + offset, 2 + offset, 65), forcing,
                           ("precipitation", "temperature", "pet"), basin, "Switzerland")


def _dates() -> DateSplitConfig:
    return DateSplitConfig(DateRange("2001-01-08", "2001-02-02"),
                           DateRange("2001-02-03", "2001-02-12"),
                           DateRange("2001-02-13", "2001-03-01"))


def test_scope_and_target_date_splits_are_leakage_free():
    series = {"a": _series("a"), "b": _series("b", .2), "target": _series("target", .4)}
    data = make_regional_hydrology_data(series,
        {key: {"height": value, "missing": None if key == "target" else value}
         for value, key in enumerate(series)},
        BasinScopeConfig(("target",), "all_except_targets"), _dates(), sequence_length=7)
    assert set(data.train.basin_ids) == {"a", "b"}
    assert set(data.validation.basin_ids) == {"target"}
    assert set(data.test.basin_ids) == {"target"}
    assert data.validation.target_dates[0] == date(2001, 2, 3)
    assert torch.allclose(data.train.inputs.reshape(-1, 6).mean(0), torch.zeros(6), atol=1e-5)


def test_all_strategies_run_on_synthetic_regional_data():
    series = {"a": _series("a"), "b": _series("b", .2), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [
        BasinCatalogRecord(key, key, {"height": index})
        for index, key in enumerate(series)
    ])
    for approach in ("soft_routing", "hard_routing", "distillation"):
        config = RegionalExperimentConfig(".", BasinScopeConfig(("target",), "exclude_targets"),
            _dates(), ModelArchitectureConfig(complex_expert="mlp", simple_expert="fourier",
                mlp_widths=(8,), fourier_frequencies=4),
            TrainingStrategyConfig(approach, 1, 1, 1, 1),
            HyperparameterConfig(sequence_length=7, batch_size=128),
            extremes=ExtremeEventConfig("statistical"))
        result = run_regional_experiment(config, catalog=catalog, series_by_basin=series)
        assert result.predictions and not result.failures
        assert set(result.per_basin_metrics) == {"target"}
        assert result.extreme_events


def test_pre_cancelled_run_returns_partial_result():
    series = {"a": _series("a"), "target": _series("target", .4)}
    catalog = CAMELSCHCatalog(".", [BasinCatalogRecord(key) for key in series])
    token = CancellationToken(); token.cancel()
    config = RegionalExperimentConfig(".", BasinScopeConfig(("target",), "exclude_targets"),
        _dates(), strategy=TrainingStrategyConfig(epochs=1),
        hyperparameters=HyperparameterConfig(sequence_length=7))
    result = run_regional_experiment(config, cancellation_token=token,
                                     catalog=catalog, series_by_basin=series)
    assert result.cancelled and not result.predictions


def test_gradio_values_map_to_the_correct_configuration_fields():
    values = [
        "2001-01-08", "2001-02-02", "2001-02-03", "2001-02-12",
        "2001-02-13", "2001-03-01", "distillation", "mlp", "fourier",
        3, 7, "4,5", 16, 0.002, 0.1, 75, 0.2, 0.3, 0.4,
        6, 7, 8, "12,10", 14, 2, 48, 9, 11, "statistical",
        0.5, 0.6, "cpu", "quantile", 0.9,
    ]
    config = _config_from_ui_values("/data", ["target"], "exclude_targets", values)
    assert config.model.mlp_widths == (12, 10)
    assert config.model.mamba_feed_forward_size == 48
    assert config.strategy.complex_teacher_epochs == 6
    assert config.hyperparameters.routing_weight == 0.5
    assert config.hyperparameters.compute_weight == 0.6
    assert config.extremes.mode == "statistical"
    assert config.extremes.value == 0.9


def test_established_basin_set_is_the_application_default():
    assert DEFAULT_BASINS == (
        "4009", "4008", "2488", "2109", "2011", "2126", "5001", "2247",
        "4022", "4023", "5009", "5010", "5016", "3014", "3015",
    )


def test_basin_map_payload_contains_geojson_and_selection_state():
    class Geometry:
        area = 2.0
        __geo_interface__ = {
            "type": "Polygon",
            "coordinates": (((7.0, 46.0), (8.0, 46.0), (8.0, 47.0), (7.0, 46.0)),),
        }

    catalog = CAMELSCHCatalog(".", [
        BasinCatalogRecord("a", "Alpha", eligible=True, geometry=Geometry()),
        BasinCatalogRecord("b", "Beta", eligible=False),
    ])
    payload = BasinMapComponent(catalog).payload(
        ["a"], active="a", training=["a"], interaction="toggle"
    )
    assert payload["selected"] == ["a"]
    assert payload["eligible"] == ["a"]
    assert payload["active"] == "a"
    assert payload["interaction"] == "toggle"
    assert payload["geojson"]["features"][0]["properties"] == {
        "basin_id": "a", "name": "Alpha", "eligible": True,
    }
    assert payload["geojson"]["features"][0]["geometry"]["type"] == "Polygon"


def test_catalog_joins_numeric_shapefile_gauge_ids(tmp_path, monkeypatch):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    observed = tmp_path / "timeseries" / "observation_based"
    simulated = tmp_path / "timeseries" / "simulation_based"
    shapes = tmp_path / "catchment_delineations"
    observed.mkdir(parents=True); simulated.mkdir(parents=True); shapes.mkdir()
    (observed / "CAMELS_CH_obs_based_4009.csv").touch()
    (simulated / "CAMELS_CH_sim_based_4009.csv").touch()
    shape_path = shapes / "CAMELS_CH_catchments.shp"
    shape_path.touch()
    frame = geopandas.GeoDataFrame(
        {"gauge_id": [4009.0]},
        geometry=[Polygon([
            (2_600_000, 1_200_000), (2_601_000, 1_200_000),
            (2_601_000, 1_201_000), (2_600_000, 1_200_000),
        ])],
        crs="EPSG:2056",
    )
    monkeypatch.setattr(geopandas, "read_file", lambda path: frame)

    record = CAMELSCHCatalog(tmp_path).get("4009")
    assert record.geometry is not None
    minx, miny, maxx, maxy = record.geometry.bounds
    assert 5 < minx < maxx < 11
    assert 45 < miny < maxy < 48


def test_regional_catalog_does_not_require_optional_annual_landcover(tmp_path):
    observed = tmp_path / "timeseries" / "observation_based"
    simulated = tmp_path / "timeseries" / "simulation_based"
    observed.mkdir(parents=True); simulated.mkdir(parents=True)
    with (observed / "CAMELS_CH_obs_based_4009.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["date", "discharge_spec(mm/d)"])
        writer.writeheader()
        for index in range(5):
            writer.writerow({"date": f"2001/01/{index + 1:02d}", "discharge_spec(mm/d)": 1 + index})
    with (simulated / "CAMELS_CH_sim_based_4009.csv").open("w", newline="") as target:
        fields = ["date", "precipitation_sim(mm/d)", "temperature_sim(degC)", "pet_sim(mm/d)"]
        writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader()
        for index in range(5):
            writer.writerow({"date": f"2001-01-{index + 1:02d}",
                "precipitation_sim(mm/d)": 2, "temperature_sim(degC)": 5,
                "pet_sim(mm/d)": .5})
    catalog = CAMELSCHCatalog(tmp_path)
    series = catalog.load_series("4009", date(2001, 1, 1), date(2001, 1, 5))
    assert len(series.dates) == 5
    assert series.forcing_names == ("precipitation", "temperature", "pet")
