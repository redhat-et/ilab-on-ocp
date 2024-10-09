.PHONY: standalone pipeline

standalone:
	python3 pipeline.py gen-standalone
	ruff format standalone/standalone.py

pipeline:
	python3 pipeline.py
