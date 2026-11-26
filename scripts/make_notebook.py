"""Write the Kaggle notebooks: kaggle/capstone_kaggle.ipynb (forecasters, UK-DALE, external test),
kaggle/habit_networks*.ipynb (the GRU and CNN-GRU as habit models), and the comparison with
Gad et al. (2025): kaggle/gad_comparison.ipynb (CPU) and kaggle/split_networks.ipynb (GPU).

    .venv\\Scripts\\python.exe scripts\\make_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASAS_URL = "https://zenodo.org/records/15708568/files/labeled_data.zip?download=1"  # = habits.casas.URL
UKDALE_URL = ("https://dap.ceda.ac.uk/edc/d1/7d78f943-f9fe-413b-af52-1816f9d968b0/data/version_0/"
              "ukdale.zip?download=1")

SETUP = """import glob, os, shutil, subprocess
# os.environ['NTFY_URL'] = 'https://ntfy.sh/<your-topic>'   # used only if the NTFY_URL secret is missing
found = glob.glob('/kaggle/input/**/habits', recursive=True)
assert found, 'No bundle: add the capstone-bundle dataset with Add Input (right-hand panel), then rerun'
bundle = os.path.dirname(found[0])
shutil.copytree(bundle, '/kaggle/working/capstone', dirs_exist_ok=True)
os.chdir('/kaggle/working/capstone')
os.makedirs('logs', exist_ok=True)
print('bundle from', bundle)
!pip install -q paho-mqtt
import torch, sklearn, pandas, sys
print('python', sys.version.split()[0], '| torch', torch.__version__, '| sklearn', sklearn.__version__,
      '| pandas', pandas.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (check the Accelerator setting)')"""

CELLS = [
    ("markdown", """# Capstone: heavy experiments on Kaggle

Runs what is too heavy for the laptop, and downloads UK-DALE at data-centre speed:

1. **REFIT, deep forecasters** (GPU): the CNNs and BiLSTMs of Yaprakdal (2023), the GRU and CNN-GRU of
   Yaprakdal (2024), with the classical forecasters and the hourly ensemble, on 20 homes.
2. **UK-DALE**: download (3.6 GB), prepare with the same rules as REFIT, then the habit models, the
   suggestions (with London's real solar) and the forecasters.
3. **External test**: models trained on REFIT (Loughborough), tested on UK-DALE (London).
4. Everything small is packed into `results.zip` (Output tab) to bring back to the laptop.

**Settings (right-hand panel):** Accelerator = GPU (T4 or P100), Internet = On (needs a phone-verified
account), and the dataset `capstone-bundle` added under Input.

**To run with the laptop closed:** Save Version > Save & Run All (Commit). Each step writes a log to
`logs/`; a failing step does not stop the others."""),
    ("code", SETUP),
    ("markdown", "## 1. REFIT: deep forecasters (GPU)"),
    ("code", """!python -m pytest -q tests/test_energy.py 2>&1 | tail -3
!python scripts/eval_forecast.py --deep 2>&1 | tee logs/refit_forecast_deep.log | tail -30"""),
    ("markdown", "## 2. UK-DALE: download and prepare"),
    ("code", f"""!mkdir -p /kaggle/temp && wget -q -c -O /kaggle/temp/ukdale.zip "{UKDALE_URL}"
!ls -l /kaggle/temp/ukdale.zip   # should be 3585155959 bytes
!CAPSTONE_DATASET=ukdale python scripts/prepare_ukdale.py /kaggle/temp/ukdale.zip 2>&1 | tee logs/ukdale_prepare.log | tail -60"""),
    ("markdown", "## 3. UK-DALE: the same experiments as on REFIT"),
    ("code", """!CAPSTONE_DATASET=ukdale python scripts/eval_refit.py 2>&1 | tee logs/ukdale_habit_models.log | tail -40
!CAPSTONE_DATASET=ukdale python scripts/eval_suggestions.py 2>&1 | tee logs/ukdale_suggestions.log | tail -40
!CAPSTONE_DATASET=ukdale python scripts/eval_forecast.py 2>&1 | tee logs/ukdale_forecast.log | tail -40
!CAPSTONE_DATASET=ukdale python scripts/eval_forecast.py --deep 2>&1 | tee logs/ukdale_forecast_deep.log | tail -30"""),
    ("markdown", "## 4. External test: trained on REFIT, tested on UK-DALE"),
    ("code", """!python scripts/eval_external.py 2>&1 | tee logs/external.log | tail -40"""),
    ("markdown", "## 5. Pack the results (download `results.zip` from the Output tab)"),
    ("code", """import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/results.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['data/results', 'data/processed/ukdale', 'data/raw/pvgis', 'logs']:
        for f in pathlib.Path(folder).rglob('*'):
            if f.is_file():
                zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/results.zip').stat().st_size / 1e6, 1), 'MB')
shutil.rmtree('/kaggle/working/capstone/data/processed/refit', ignore_errors=True)  # keep the Output small"""),
]


def habit_cells(args: str = "", tag: str = "") -> list:
    """The habit-network notebook. Run 1: no args. Run 2: '--weeks 4 --epochs 60', tag '_4w'."""
    csv = f"habit_networks{tag}"
    name = "Habit networks" + (f" ({tag.strip('_')})" if tag else "")
    run = f"python -u scripts/eval_habit_networks.py {args}".strip()
    intro = ("the networks read the raw past week hour by hour" if not args else
             f"run 2 ({args}), fixed before seeing its results: the networks get the same 4 weeks of "
             "history as the forest's features, and the GRU is not cut off by the epoch cap")
    return [
        ("markdown", f"""# Capstone: the professor's GRU and CNN-GRU as habit models{' - ' + tag.strip('_') if tag else ''}

"Will this appliance be switched on in the next hour?", the same question, split and scoring as the
random forest, on 123 REFIT and 54 UK-DALE tasks; {intro}.

**Settings:** Accelerator = GPU, dataset `capstone-bundle` (latest version) under Input, Internet = On
(only for the phone notifications). Save Version > Save & Run All (Commit). Progress streams to the
log, one line per task.

**Phone notifications (optional):** add a Kaggle secret `NTFY_URL` = `https://ntfy.sh/<your topic>`
(Add-ons > Secrets, attached to this notebook); if Kaggle keeps dropping it, uncomment the first
`os.environ['NTFY_URL'] = ...` line instead. You get: started, progress every 30 min, each step
done or failed, finished.

**A new notebook copies none of these settings:** add the input, GPU, Internet (and the topic) each time."""),
        ("code", SETUP + f"""
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU: check the Accelerator setting'
notify('{name}: started', f'GPU: {{gpu}}\\nREFIT 123 tasks, then UK-DALE 54; x2 networks {args}',
       'default' if torch.cuda.is_available() else 'high', 'rocket')"""),
        ("code", f"""ok = run_step('Tests', 'python -m pytest -q tests/test_habit_networks.py', 'logs/tests.log', summary_lines=2)
ok &= run_step('REFIT {name}', '{run}', 'logs/refit_{csv}.log')"""),
        ("code", f"""ok &= run_step('UK-DALE {name}', '{run}', 'logs/ukdale_{csv}.log', env={{'CAPSTONE_DATASET': 'ukdale'}})"""),
        ("code", f"""import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/{csv}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in ['data/results/{csv}.csv', 'data/results/ukdale/{csv}.csv',
              'logs/refit_{csv}.log', 'logs/ukdale_{csv}.log']:
        if pathlib.Path(f).exists():
            zf.write(f, f)
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
print(sorted(os.listdir('/kaggle/working')))
notify('{name}: FINISHED' if ok else '{name}: finished WITH ERRORS',
       '{csv}.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


CASAS_CELL = f"""!mkdir -p /kaggle/temp && wget -q -c -O /kaggle/temp/casas.zip "{CASAS_URL}"
!ls -l /kaggle/temp/casas.zip   # should be 236037656 bytes
ok = run_step('CASAS prepare', 'CAPSTONE_DATASET=casas python -u scripts/prepare_casas.py /kaggle/temp/casas.zip',
              'logs/casas_prepare.log', summary_lines=3)
casas_ok = ok  # without CASAS, its steps are skipped (REFIT and UK-DALE still run)"""
DATASETS_LINE = "DATASETS = ['refit', 'ukdale', 'casas']   # to rerun only CASAS: ['casas']"


def comparison_cells(gpu: bool) -> list:
    """gad_comparison (CPU): replication on CASAS, our pipeline on CASAS, the split effect with the
    trees on all three datasets. split_networks (GPU): the GRU under both splits."""
    name = "Split effect, GRU" if gpu else "Gad comparison"
    out = "split_networks" if gpu else "gad_comparison"
    if gpu:
        intro = """The professor's GRU under both splits (by time, and at random as Gad et al. split), on REFIT,
UK-DALE and CASAS: do networks look better than trees only because of the split?

**Settings:** Accelerator = **GPU**, Internet = **On** (CASAS download and notifications), dataset
`capstone-bundle` (latest version) under Input. Save Version > Save & Run All (Commit)."""
        steps = DATASETS_LINE + """
ok &= run_step('Tests', 'python -m pytest -q tests/test_gad.py tests/test_habit_networks.py', 'logs/tests.log', summary_lines=2)
for ds in DATASETS:
    if ds != 'casas' or casas_ok:
        ok &= run_step(f'{ds} GRU, both splits', 'python -u scripts/eval_split_effect.py --models GRU',
                       f'logs/{ds}_split_gru.log', env={'CAPSTONE_DATASET': ds})"""
    else:
        intro = """The comparison article, Gad et al. (2025, *Sensors* 25(19):6082), both ways:

1. **Replication** on their data: 21 CASAS homes, their features, LightGBM / XGBoost / CatBoost with
   their Table 3 settings, their random split; then the same with a time split, then with only what a
   home knows (`scripts/eval_gad.py`).
2. **Our method on their data**: our habit models on CASAS ("will this activity start in this hour?").
3. **Their method on our data**, and **the split effect**: baseline, random forest and their LightGBM on
   REFIT, UK-DALE and CASAS, split by time and at random (`scripts/eval_split_effect.py`).

**Settings:** Accelerator = **None** (CPU is enough; no GPU queue), Internet = **On** (downloads CASAS,
236 MB, from Zenodo), dataset `capstone-bundle` (latest version) under Input.
Save Version > Save & Run All (Commit). Phone notifications as before (secret `NTFY_URL`)."""
        steps = DATASETS_LINE + """
ok &= run_step('Tests', 'python -m pytest -q tests/test_gad.py', 'logs/tests.log', summary_lines=2)
if 'casas' in DATASETS and casas_ok:
    ok &= run_step('Gad replication (CASAS)', 'python -u scripts/eval_gad.py', 'logs/casas_gad.log',
                   env={'CAPSTONE_DATASET': 'casas'}, summary_lines=40)
    ok &= run_step('Our habit models on CASAS', 'python -u scripts/eval_refit.py', 'logs/casas_habit_models.log',
                   env={'CAPSTONE_DATASET': 'casas'}, summary_lines=30)
for ds in DATASETS:
    if ds != 'casas' or casas_ok:
        ok &= run_step(f'{ds} split effect', 'python -u scripts/eval_split_effect.py',
                       f'logs/{ds}_split_effect.log', env={'CAPSTONE_DATASET': ds}, summary_lines=20)"""
    return [
        ("markdown", f"# Capstone: {name}\n\n{intro}\n\n**A new notebook copies none of these settings.**"),
        ("code", SETUP + f"""
import lightgbm, xgboost, catboost
print('lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| catboost', catboost.__version__)
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
notify('{name}: started', 'CASAS download, then the runs', 'default', 'rocket')"""),
        ("code", CASAS_CELL),
        ("code", steps),
        ("code", f"""import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/{out}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in [*pathlib.Path('data/results').rglob('*'), *pathlib.Path('logs').rglob('*')]:
        if f.is_file() and (f.suffix == '.log' or f.name.startswith(('split_effect', 'gad_')) or 'casas' in f.parts):
            zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/{out}.zip').stat().st_size / 1e6, 2), 'MB')
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
notify('{name}: FINISHED' if ok else '{name}: finished WITH ERRORS',
       '{out}.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


def forecast_cells():
    """Her networks with the same chances as the trees (scripts/eval_forecast_habits.py --deep), GPU."""
    name = out = "forecast_networks"
    return [
        ("markdown", f"""# Capstone: {name}

Fairness check for the load forecast: gradient boosting with the home's own information beat
persistence and her networks. Here her GRU, CNN-GRU and BiLSTM (RNN2) get the same chances: the
absolute-error loss as well as hers, and the habit forecasts and appliance states as extra hourly
channels. REFIT (20 homes) then UK-DALE (5 homes), same test hours as before.

**Settings:** Accelerator = **GPU T4**, Internet = **On** (London weather from PVGIS), dataset
`capstone-bundle` (latest version) under Input. Save Version > Save & Run All (Commit).

**A new notebook copies none of these settings.**"""),
        ("code", SETUP + f"""
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
notify('{name}: started', 'REFIT then UK-DALE', 'default', 'rocket')"""),
        ("code", """ok = run_step('Tests', 'python -m pytest -q tests/test_energy.py', 'logs/tests.log', summary_lines=2)
ok &= run_step('REFIT networks, same chances', 'python -u scripts/eval_forecast_habits.py --deep',
               'logs/refit_forecast_networks.log', summary_lines=30)
ok &= run_step('UK-DALE networks, same chances', 'python -u scripts/eval_forecast_habits.py --deep',
               'logs/ukdale_forecast_networks.log', env={'CAPSTONE_DATASET': 'ukdale'}, summary_lines=30)"""),
        ("code", f"""import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/{out}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in [*pathlib.Path('data/results').rglob('forecast_habits_deep.csv'), *pathlib.Path('logs').rglob('*.log')]:
        zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/{out}.zip').stat().st_size / 1e6, 2), 'MB')
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
notify('{name}: FINISHED' if ok else '{name}: finished WITH ERRORS',
       '{out}.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


def casas_analysis_cells():
    """CASAS described for the dataset analysis (scripts/analyze_casas.py), CPU, a few minutes."""
    name = out = "casas_analysis"
    return [
        ("markdown", f"""# Capstone: {name}

Describes the CASAS data (the followed article's) for the dataset analysis: when the common activities
start, how often, and how often an activity repeats from one sensor interval to the next. Downloads
CASAS (236 MB) and prepares it, then writes a small summary file.

**Settings:** Accelerator = **None** (CPU), Internet = **On**, dataset `capstone-bundle` (latest version)
under Input. Save Version > Save & Run All (Commit).

**A new notebook copies none of these settings.**"""),
        ("code", SETUP + f"""
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
notify('{name}: started', 'CASAS download, prepare, summary', 'default', 'rocket')"""),
        ("code", CASAS_CELL + """
if casas_ok:
    ok &= run_step('CASAS summary', 'python -u scripts/analyze_casas.py', 'logs/casas_analysis.log',
                   env={'CAPSTONE_DATASET': 'casas'}, summary_lines=5)"""),
        ("code", f"""import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/{out}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in [*pathlib.Path('data/results').rglob('casas_summary.json'), *pathlib.Path('logs').rglob('*.log')]:
        zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/{out}.zip').stat().st_size / 1e6, 3), 'MB')
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
notify('{name}: FINISHED' if ok else '{name}: finished WITH ERRORS',
       '{out}.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


def xai_cells():
    """xai_boosting (CPU): gradient boosting as habit models next to the forest, SHAP for the habit
    models and for the load forecaster, on REFIT, UK-DALE and CASAS."""
    intro = """Two questions, one run (CPU):

1. **Is the random forest the best tree model for habits?** LightGBM and CatBoost, with the same 15
   inputs, settings fixed in advance, and the same threshold rule, next to the baseline, logistic
   regression and the forest, on REFIT, UK-DALE and CASAS (`scripts/eval_refit.py --models ...`).
2. **Explainable AI (SHAP).** Which inputs drive each habit model's predictions
   (`scripts/explain_habits.py`, for the forest, LightGBM and CatBoost), and what drives the
   gradient-boosting load forecaster (`scripts/explain_forecast.py`, REFIT and UK-DALE), and why the
   forest guard restored power on real test nights, for rescues and false alarms
   (`scripts/explain_guard.py`, REFIT and UK-DALE).

**Settings:** Accelerator = **None** (CPU), Internet = **On** (CASAS from Zenodo, London's solar from
PVGIS), dataset `capstone-bundle` (latest version) under Input. Save Version > Save & Run All (Commit).
Phone notifications as before (secret `NTFY_URL`). Expect a few hours; each step logs to `logs/`."""
    steps = DATASETS_LINE + """
MODELS = ['random_forest', 'lightgbm', 'catboost']
ok &= run_step('Tests', 'python -m pytest -q tests/test_features.py tests/test_habits.py tests/test_explain.py', 'logs/tests.log', summary_lines=2)
for ds in DATASETS:
    if ds == 'casas' and not casas_ok:
        continue
    env = {'CAPSTONE_DATASET': ds}
    ok &= run_step(f'{ds} habit models + boosting', 'python -u scripts/eval_refit.py --models lightgbm,catboost',
                   f'logs/{ds}_habit_boosting.log', env=env, summary_lines=30)
    for m in MODELS:
        ok &= run_step(f'{ds} SHAP {m}', f'python -u scripts/explain_habits.py --model {m}',
                       f'logs/{ds}_shap_{m}.log', env=env, summary_lines=25)
    if ds != 'casas':  # CASAS has no electricity load to forecast, nor standby cuts
        ok &= run_step(f'{ds} SHAP forecaster', 'python -u scripts/explain_forecast.py',
                       f'logs/{ds}_shap_forecast.log', env=env, summary_lines=15)
        ok &= run_step(f'{ds} guard reasons', 'python -u scripts/explain_guard.py',
                       f'logs/{ds}_guard_reasons.log', env=env, summary_lines=40)"""
    return [
        ("markdown", f"# Capstone: boosting and explainable AI\n\n{intro}\n\n**A new notebook copies none of these settings.**"),
        ("code", SETUP + """
import lightgbm, catboost, shap
print('lightgbm', lightgbm.__version__, '| catboost', catboost.__version__, '| shap', shap.__version__)
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
notify('Boosting + XAI: started', 'CASAS download, then the runs', 'default', 'rocket')"""),
        ("code", CASAS_CELL),
        ("code", steps),
        ("code", """import zipfile, pathlib
keep = ('habit_models', 'habit_shap', 'forecast_shap', 'guard_reasons')
with zipfile.ZipFile('/kaggle/working/xai_boosting.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in pathlib.Path('data/results').rglob('*.csv'):
        if f.name.startswith(keep):
            zf.write(f, str(f))
    for f in [*pathlib.Path('data/results').rglob('figures/xai-*.png'), *pathlib.Path('logs').rglob('*.log')]:
        zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/xai_boosting.zip').stat().st_size / 1e6, 2), 'MB')
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
notify('Boosting + XAI: FINISHED' if ok else 'Boosting + XAI: finished WITH ERRORS',
       'xai_boosting.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


def weekly_cells():
    """weekly_habits (CPU): CatBoost finds habits every week from the last 3 weeks, against counting."""
    intro = """**Weekly habit finding with a model** (`scripts/eval_weekly_habits.py`, `habits/weekly.py`): every week,
CatBoost predicts from the last 3 weeks which slots will be habits next week, per appliance (so one slot
can hold several habits), and keeps or drops existing ones (two thresholds). Its inputs include the other
appliances in four ways (same slot, order, together or apart, room groups), tested as an ablation. Against
counting once (today's system), counting over the last 3 weeks, and the last 3 weeks' rate; each home is
predicted by a model trained on the other homes. Then SHAP by input group.

**Settings:** Accelerator = **None** (CPU), Internet = **On** (CASAS from Zenodo), dataset `capstone-bundle`
(latest version) under Input. Save Version > Save & Run All (Commit)."""
    steps = DATASETS_LINE + """
ok &= run_step('Tests', 'python -m pytest -q tests/test_weekly.py', 'logs/tests.log', summary_lines=2)
for ds in DATASETS:
    if ds == 'casas' and not casas_ok:
        continue
    ok &= run_step(f'{ds} weekly habits', 'python -u scripts/eval_weekly_habits.py',
                   f'logs/{ds}_weekly_habits.log', env={'CAPSTONE_DATASET': ds}, summary_lines=40)"""
    return [
        ("markdown", f"# Capstone: weekly habit finding\n\n{intro}\n\n**A new notebook copies none of these settings.**"),
        ("code", SETUP + """
import catboost, shap, warnings
warnings.filterwarnings('ignore', message='LightGBM binary classifier')
print('catboost', catboost.__version__, '| shap', shap.__version__)
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
notify('Weekly habits: started', 'CASAS download, then the runs', 'default', 'rocket')"""),
        ("code", CASAS_CELL),
        ("code", steps),
        ("code", """import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/weekly_habits.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in [*pathlib.Path('data/results').rglob('weekly_habits*.csv'), *pathlib.Path('logs').rglob('*.log')]:
        zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/weekly_habits.zip').stat().st_size / 1e6, 2), 'MB')
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
notify('Weekly habits: FINISHED' if ok else 'Weekly habits: finished WITH ERRORS',
       'weekly_habits.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


def weekly_models_cells():
    """weekly_models (CPU): training regimes (own, other, hybrid) and six models for weekly habit finding."""
    intro = """**Weekly habit finding: which data and which model** (`scripts/eval_weekly_models.py`). (1) CatBoost
trained on the home's own past only, on the other homes only, and on both (hybrid). (2) In the hybrid
regime, six models: random forest, XGBoost, LightGBM, CatBoost (tree ensembles: bagging and the three
boosting libraries of Gad et al.), a linear model and a neural network (MLP). Same examples, decisions and
scoring as `eval_weekly_habits.py`; thresholds never see the test weeks.

**Settings:** Accelerator = **None** (CPU), Internet = **On** (CASAS from Zenodo), dataset `capstone-bundle`
(latest version) under Input. Save Version > Save & Run All (Commit)."""
    steps = DATASETS_LINE + """
ok &= run_step('Tests', 'python -m pytest -q tests/test_weekly.py', 'logs/tests.log', summary_lines=2)
for ds in DATASETS:
    if ds == 'casas' and not casas_ok:
        continue
    ok &= run_step(f'{ds} weekly regimes + models', 'python -u scripts/eval_weekly_models.py',
                   f'logs/{ds}_weekly_models.log', env={'CAPSTONE_DATASET': ds}, summary_lines=45)"""
    return [
        ("markdown", f"# Capstone: weekly habits, regimes and models\n\n{intro}\n\n**A new notebook copies none of these settings.**"),
        ("code", SETUP + """
import catboost, xgboost, lightgbm, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
print('catboost', catboost.__version__, '| xgboost', xgboost.__version__, '| lightgbm', lightgbm.__version__)
sys.path.insert(0, 'scripts')
from kaggle_steps import ADDRESS, notify, run_step
print('ntfy:', 'on' if ADDRESS else 'off (no NTFY_URL secret or first-line topic)')
notify('Weekly models: started', 'CASAS download, then the runs', 'default', 'rocket')"""),
        ("code", CASAS_CELL),
        ("code", steps),
        ("code", """import zipfile, pathlib
with zipfile.ZipFile('/kaggle/working/weekly_models.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in [*pathlib.Path('data/results').rglob('weekly_models*.csv'), *pathlib.Path('logs').rglob('*.log')]:
        zf.write(f, str(f))
print(round(pathlib.Path('/kaggle/working/weekly_models.zip').stat().st_size / 1e6, 2), 'MB')
shutil.rmtree('/kaggle/working/capstone', ignore_errors=True)  # keep the Output to the zip
notify('Weekly models: FINISHED' if ok else 'Weekly models: finished WITH ERRORS',
       'weekly_models.zip is in the Output tab', 'default' if ok else 'high', 'tada' if ok else 'warning')"""),
    ]


NOTEBOOKS = {"capstone_kaggle.ipynb": CELLS, "habit_networks.ipynb": habit_cells(),
             "habit_networks_4w.ipynb": habit_cells("--weeks 4 --epochs 60", "_4w"),
             "gad_comparison.ipynb": comparison_cells(gpu=False), "split_networks.ipynb": comparison_cells(gpu=True),
             "forecast_networks.ipynb": forecast_cells(),
             "casas_analysis.ipynb": casas_analysis_cells(), "xai_boosting.ipynb": xai_cells(),
             "weekly_habits.ipynb": weekly_cells(), "weekly_models.ipynb": weekly_models_cells()}


def write(name: str, spec) -> None:
    cells = []
    for kind, text in spec:
        lines = text.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "kaggle" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(out)


def main() -> None:
    for name, spec in NOTEBOOKS.items():
        write(name, spec)


if __name__ == "__main__":
    main()
