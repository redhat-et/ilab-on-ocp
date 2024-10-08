.PHONY: standalone

standalone:
	python3 pipeline.py gen-standalone
	ruff format standalone/standalone.py
