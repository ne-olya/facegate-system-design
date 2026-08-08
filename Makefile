demo:
	cd poc && python3 run_demo.py

test:
	cd poc && python3 -m pytest -q

.PHONY: demo test
