"""Cost-aware smart start: the cheapest start within the deadline, and the resident's quiet hours."""

from datetime import datetime

from home.scheduler import in_quiet_hours, make_plan, run_cost

NO_SUN = lambda t: 0.0  # noqa: E731
NOON_SUN = lambda t: 3000.0 if 11 <= t.hour < 15 else 0.0  # noqa: E731
NIGHT_CHEAP = lambda t: 9.44 if 1 <= t.hour < 7 else 16.88  # noqa: E731 (an Economy 7-like tariff)


def test_cost_counts_grid_at_import_and_used_solar_at_export():
    # 1 kW for 60 minutes: 1 kWh. No sun at 16.88 p -> 16.88 p; full sun at 5.5 p export -> 5.5 p.
    assert round(run_cost(datetime(2014, 3, 3, 20), 60, 1000, NO_SUN, NIGHT_CHEAP), 2) == 16.88
    assert round(run_cost(datetime(2014, 3, 3, 12), 60, 1000, NOON_SUN, NIGHT_CHEAP, 5.5), 2) == 5.5


def test_without_a_panel_it_picks_the_cheap_night():
    plan = make_plan("dishwasher", datetime(2014, 3, 3, 21), datetime(2014, 3, 4, 18), 1000, 120,
                     NO_SUN, price=NIGHT_CHEAP)
    assert 1 <= plan.chosen.hour < 7
    assert plan.options[0].cost_pence < 2 * 16.88


def test_quiet_hours_move_the_start_out_of_the_night():
    plan = make_plan("dishwasher", datetime(2014, 3, 3, 21), datetime(2014, 3, 4, 18), 1000, 120,
                     NO_SUN, price=NIGHT_CHEAP, quiet=(22, 7))
    assert not in_quiet_hours(plan.chosen, (22, 7))
    assert all(not in_quiet_hours(o.start, (22, 7)) for o in plan.options)


def test_the_sun_beats_the_cheap_night_when_export_pays_less():
    plan = make_plan("dishwasher", datetime(2014, 3, 3, 21), datetime(2014, 3, 4, 18), 1000, 120,
                     NOON_SUN, price=NIGHT_CHEAP, export_price=5.5)
    assert 11 <= plan.chosen.hour < 15


def test_without_a_price_it_still_ranks_by_solar():
    plan = make_plan("dishwasher", datetime(2014, 3, 3, 8), datetime(2014, 3, 3, 18), 1000, 120, NOON_SUN)
    assert plan.chosen.hour == 11 and plan.options[0].cost_pence is None


def test_quiet_hours_across_midnight_and_within_a_day():
    assert in_quiet_hours(datetime(2014, 3, 3, 23), (22, 7)) and in_quiet_hours(datetime(2014, 3, 3, 3), (22, 7))
    assert not in_quiet_hours(datetime(2014, 3, 3, 7), (22, 7))
    assert in_quiet_hours(datetime(2014, 3, 3, 13), (12, 14)) and not in_quiet_hours(datetime(2014, 3, 3, 14), (12, 14))
    assert not in_quiet_hours(datetime(2014, 3, 3, 3), None)
