"""Why the habit model predicts what it does, in words a resident can read (explainable AI).

The habit model (habits/models.py, habits/forest.py) answers "will this appliance be switched on
in this hour?" from 15 inputs (habits/features.py). SHAP (TreeSHAP, the `shap` library) splits one
prediction into one contribution per input; inputs that only mean something together (the hour's
sine and cosine, the weekday's) are summed into one group. The groups that push the prediction most
towards a switch-on become the reasons, phrased with the hour's own values:

    "used at this hour yesterday; last used 2 hours ago; 3 other appliances switched on in the last 3 hours"

Used in two places of the running home:
  - the forest guard (habits/forest.py) says why it restored a cut plug (phone note, dashboard);
  - the forest's check of a suggestion says what its agreement rests on (generic labels, averaged
    over the hours of the window on the check days).
scripts/explain_habits.py uses the same groups for the paper's figures.
"""

from dataclasses import dataclass

import pandas as pd

GROUPS = {  # input group -> the model's columns (habits/features.py)
    "Hour of day": ["hour_sin", "hour_cos"],
    "Day of week": ["dow_sin", "dow_cos"],
    "Weekend": ["weekend"],
    "Used in the previous hour": ["lag_1h"],
    "Used at this hour yesterday": ["lag_24h"],
    "Used at this hour a week ago": ["lag_168h"],
    "Hours with a use, last 24 h": ["uses_24h"],
    "Hours with a use, last 7 days": ["uses_7d"],
    "Hours since the last use": ["hours_since_use"],
    "Share of the last 4 weeks used at this hour": ["recent_how_rate"],
    "Rest of the house active, last hour": ["others_1h"],
    "Rest of the house active, last 3 hours": ["others_3h"],
    "Previous hour had data": ["prev_hour_covered"],
}
NOT_A_REASON = {"Previous hour had data"}  # a data-quality input: true, but no reason to give a person

GENERIC = {  # what a group means, without the hour's values (for reasons averaged over many hours)
    "Hour of day": "the time of day",
    "Day of week": "the day of the week",
    "Weekend": "weekday or weekend",
    "Used in the previous hour": "use in the hour before",
    "Used at this hour yesterday": "use at this hour the day before",
    "Used at this hour a week ago": "use at this hour a week before",
    "Hours with a use, last 24 h": "how much it was used that day",
    "Hours with a use, last 7 days": "how much it was used that week",
    "Hours since the last use": "how recently it was used",
    "Share of the last 4 weeks used at this hour": "its use at this hour in recent weeks",
    "Rest of the house active, last hour": "activity elsewhere in the house",
    "Rest of the house active, last 3 hours": "activity elsewhere in the house",
}
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass
class Reason:
    group: str
    phrase: str
    contribution: float  # SHAP value; positive pushes towards a switch-on


def grouped(values, columns) -> pd.DataFrame:
    """Sum each group's columns (signed), row by row: one contribution per input group."""
    frame = pd.DataFrame(values, columns=list(columns))
    return pd.DataFrame({g: frame[cols].sum(axis=1) for g, cols in GROUPS.items() if set(cols) <= set(frame)})


def phrase(group: str, row: pd.Series, when: pd.Timestamp) -> str:
    """The reason in words, with this hour's values."""
    n = lambda column: int(round(float(row[column])))  # noqa: E731
    if group == "Hour of day":
        return f"it is {when:%H}:00"
    if group == "Day of week":
        return f"it is a {WEEKDAY_NAMES[when.dayofweek]}"
    if group == "Weekend":
        return "it is the weekend" if row["weekend"] else "it is a weekday"
    if group in ("Used in the previous hour", "Used at this hour yesterday", "Used at this hour a week ago"):
        column = GROUPS[group][0]
        return group[0].lower() + group[1:] if row[column] else "not " + group[0].lower() + group[1:]
    if group == "Hours with a use, last 24 h":
        return f"used in {n('uses_24h')} of the last 24 hours"
    if group == "Hours with a use, last 7 days":
        return f"used in {n('uses_7d')} hours over the last 7 days"
    if group == "Hours since the last use":
        h = n("hours_since_use")
        return "not used for a week or more" if h >= 168 else f"last used {h} hour{'s' if h != 1 else ''} ago"
    if group == "Share of the last 4 weeks used at this hour":
        return f"used at this hour in {float(row['recent_how_rate']):.0%} of the last 4 weeks"
    if group == "Rest of the house active, last hour":
        return f"{n('others_1h')} other appliance switch-ons in the last hour"
    if group == "Rest of the house active, last 3 hours":
        return f"{n('others_3h')} other appliance switch-ons in the last 3 hours"
    return group


class Explainer:
    """TreeSHAP for one tree model (random forest, LightGBM, CatBoost), created once and reused."""

    def __init__(self, model):
        import shap  # imported only when an explanation is asked for

        self._tree = shap.TreeExplainer(model)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        values = self._tree.shap_values(X)
        values = values[:, :, 1] if getattr(values, "ndim", 2) == 3 else values  # towards "switched on"
        out = grouped(values, X.columns)
        out.index = X.index
        return out

    def reasons(self, X_hour: pd.DataFrame, top: int = 3) -> list[Reason]:
        """The groups pushing this one hour's prediction most towards a switch-on, in words."""
        c = self.contributions(X_hour).iloc[0].drop(list(NOT_A_REASON), errors="ignore")
        best = c[c > 0].sort_values(ascending=False).head(top)
        row, when = X_hour.iloc[0], X_hour.index[0]
        return [Reason(g, phrase(g, row, when), float(v)) for g, v in best.items()]

    def main_reasons(self, X: pd.DataFrame, towards_use: bool = True, top: int = 2) -> list[str]:
        """What a verdict over many hours mostly rests on, in generic words (no single hour's values)."""
        c = self.contributions(X).drop(columns=list(NOT_A_REASON), errors="ignore")
        c = c.clip(lower=0) if towards_use else (-c).clip(lower=0)
        labels = []
        for g in c.mean().sort_values(ascending=False).index:
            if GENERIC[g] not in labels and c[g].mean() > 0:
                labels.append(GENERIC[g])
            if len(labels) == top:
                break
        return labels


def sentence(reasons: list[Reason]) -> str:
    return "; ".join(r.phrase for r in reasons)
