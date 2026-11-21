from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.config import LOCALITIES
from app.services import aq_data, aq_hotspots, aq_sim, forecast, sources
from app.services.geo import offset_latlon
from scripts import fetch_open_meteo as fom


@pytest.fixture(scope="module")
def sim():
    t0 = pd.Timestamp("2026-11-04 12:00")
    return aq_sim.simulate(t0, days_back=20, hours_ahead=72, seed=3), t0


def _incident(lat, lon, cat, priority=80, status="assigned", verified=True, supporters=3, reports=2):
    return SimpleNamespace(lat=lat, lon=lon, category=cat, priority_score=priority, status=status, verified=verified,
                           unique_supporters=supporters, report_count=reports)


def test_pm25_bands():
    cats = {v: aq_data.pm25_category(v)[0] for v in (30, 31, 60, 90, 120, 121, 250, 251)}
    assert cats == {30: "Good", 31: "Satisfactory", 60: "Satisfactory", 90: "Moderate", 120: "Poor",
                    121: "Very poor", 250: "Very poor", 251: "Severe"}


def test_cardinal():
    assert [aq_data.cardinal(d) for d in (0, 90, 225, 359)] == ["N", "E", "SW", "N"]


def test_simulator_structure_and_reproducibility(sim):
    df, t0 = sim
    assert set(df.location) == set(LOCALITIES)
    assert df.isna().sum().sum() == 0 and df.pm25.between(8, 700).all()
    g = df[df.location == "Okhla"].sort_values("time")
    assert (g.time.diff().dropna() == pd.Timedelta(hours=1)).all() and g.time.max() == t0 + pd.Timedelta(hours=72)
    again = aq_sim.simulate(t0, days_back=20, hours_ahead=72, seed=3)
    assert df.equals(again) and not df.equals(aq_sim.simulate(t0, days_back=20, hours_ahead=72, seed=4))


def test_open_meteo_parser_schema():
    t = ["2026-11-04T00:00", "2026-11-04T01:00"]
    aq = {"hourly": {"time": t, "pm2_5": [100.0, 110.0], "pm10": [180.0, 190.0], "nitrogen_dioxide": [40.0, 42.0]}}
    wx = {"hourly": {"time": t, "temperature_2m": [18, 17], "relative_humidity_2m": [70, 72], "wind_speed_10m": [5, 6],
                     "wind_direction_10m": [300, 310], "precipitation": [0, 0], "surface_pressure": [1010, 1011]}}
    df = fom.parse(aq, wx, "Okhla", 28.5, 77.2)
    assert list(df.columns) == fom.COLUMNS and df.pm25.tolist() == [100.0, 110.0]
    assert np.issubdtype(df.time.dtype, np.datetime64) and (df.location == "Okhla").all()


def test_first_crossing():
    fc = pd.DataFrame({"time": pd.date_range("2026-11-04", periods=4, freq="h"), "pm25": [80, 130, 170, 90]})
    r = forecast.first_crossing(fc, 120)
    assert r["first_value"] == 130 and r["peak_value"] == 170
    assert forecast.first_crossing(fc, 300) is None


@pytest.mark.skipif(not forecast.models_available(), reason="run `python -m ml.train_forecast` first")
class TestForecastModel:
    def test_forecast_shape_and_band(self):
        df, meta = aq_data.load_air_quality()
        fc = forecast.forecast_all(df, meta["t0"])
        assert len(fc) == len(LOCALITIES) * 48
        assert (fc.lo <= fc.pm25 + 1e-6).all() and (fc.pm25 <= fc.hi + 1e-6).all() and (fc.pm25 >= 0).all()
        assert fc.time.min() == meta["t0"] + pd.Timedelta(hours=1)

    def test_beats_baselines_and_band_is_calibrated(self):
        m = forecast.load_metrics()
        assert m["overall"]["rmse"] < m["persistence"]["rmse"] and m["overall"]["rmse"] < m["seasonal_naive"]["rmse"]
        assert 0.65 < m["band_coverage_80"] < 0.95
        assert abs(sum(m["importance"].values()) - 1) < 0.01


def test_hotspot_table_ranking_and_pressure(sim):
    df, t0 = sim
    base = aq_hotspots.station_table(df, t0, [])
    assert base.hotspot_index.is_monotonic_decreasing and base.hotspot_index.between(0, 100).all()
    lat, lon, _ = LOCALITIES["Rohini"]
    more = aq_hotspots.station_table(df, t0, [_incident(lat, lon, "garbage_burning", 95)])
    pick = lambda t: float(t.loc[t.location == "Rohini", "hotspot_index"].iloc[0])
    assert pick(more) > pick(base)
    assert int(more.loc[more.location == "Rohini", "open_incidents"].iloc[0]) == 1
    resolved = aq_hotspots.station_table(df, t0, [_incident(lat, lon, "garbage_burning", 95, status="resolved")])
    assert pick(resolved) == pick(base)           # resolved incidents no longer add pressure


def test_source_shares_and_citizen_evidence(sim):
    df, t0 = sim
    none = sources.attribute_sources(df, t0, [])
    assert all(abs(sum(v["shares"].values()) - 1) < 1e-9 for v in none.values())
    lat, lon, _ = LOCALITIES["Dwarka"]
    with_reports = sources.attribute_sources(df, t0, [_incident(lat, lon, "garbage_burning", 90)] * 3)
    assert with_reports["Dwarka"]["shares"]["garbage_burning"] > none["Dwarka"]["shares"]["garbage_burning"]
    assert with_reports["Dwarka"]["n_incidents"] == 3
    assert with_reports["Okhla"]["shares"] == none["Okhla"]["shares"]       # far-away reports don't leak


def test_industrial_area_prior_dominates_without_reports(sim):
    df, t0 = sim
    shares = sources.attribute_sources(df, t0, [])["Okhla"]["shares"]
    assert max(shares, key=shares.get) == "industrial_smoke"


def test_emission_grid_peaks_near_reports():
    lat, lon, _ = LOCALITIES["Okhla"]
    pts = pd.DataFrame([(lat, lon, "industrial_smoke", 1.0)] * 5, columns=["lat", "lon", "category", "weight"])
    G = sources.emission_grids(pts, bandwidth_km=2.0)
    grid = G["grids"]["industrial_smoke"]
    iy, ix = np.unravel_index(grid.argmax(), grid.shape)
    lon0, lon1, lat0, lat1 = G["extent"]
    peak_lat, peak_lon = lat0 + (lat1 - lat0) * iy / (grid.shape[0] - 1), lon0 + (lon1 - lon0) * ix / (grid.shape[1] - 1)
    from app.services.geo import haversine_m
    assert float(haversine_m(lat, lon, peak_lat, peak_lon)) < 1500
    assert G["grids"]["garbage_burning"].max() == 0
