PY ?= .venv/bin/python
export MPLCONFIGDIR ?= /private/tmp/mpl-jump-risk

.PHONY: help venv test backtest report manifest scorecard sensitivity figures daily clean

help:
	@echo "make venv         create .venv and install dependencies (Python >= 3.10)"
	@echo "make test         run the validation suite            (~4 min)"
	@echo "make backtest     walk-forward, writes walkforward.csv (~8 min)"
	@echo "make report       scorecard + PSD diagnostic + traffic light + figures"
	@echo "make manifest     write the reproducibility manifest with SHA-256 hashes"
	@echo "make sensitivity  bounded lambda x window grid         (~7 min)"
	@echo "make daily        one shadow-mode run with persistent state"
	@echo ""
	@echo "report/scorecard need backtest to have run first (walkforward.csv)."

venv:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e ".[data]"
	@$(PY) -c "import sys; assert sys.version_info >= (3,10), 'requires Python >= 3.10'; print('ready:', sys.version.split()[0])"

test:
	$(PY) tests/test_core.py

backtest:
	$(PY) -m src.backtest

scorecard:
	$(PY) -m src.scorecard

sensitivity:
	$(PY) -m src.sensitivity

figures:
	$(PY) -m src.plots

report: scorecard figures
	@echo ""
	@$(PY) -m src.basel
	@echo ""
	@$(PY) -c "from src import data, merton; \
	rets = data.log_returns(); \
	p = {c: merton.calibrar(rets[c].values) for c in rets.columns}; \
	print('DIAGNOSTICO PSD DE LA COVARIANZA DE DIFUSION'); \
	print(merton.reporte_psd(rets, p).to_string(index=False))"

manifest:
	$(PY) -m src.provenance --write >/dev/null && $(PY) -m src.provenance --check

daily:
	$(PY) -m src.diario

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
