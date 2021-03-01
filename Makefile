# Convenience wrapper; every target is a thin call into CMake / Python.
PY      ?= .venv/bin/python
BUILD   ?= build
JOBS    ?= $(shell nproc)

.PHONY: all venv model build test test-python test-tsan test-asan bench run bench-compare docker kind clean

all: model build

venv:
	python3 -m venv .venv
	.venv/bin/pip install -q -r training/requirements.txt cmake

model: models/model.onnx
models/model.onnx: training/train.py training/telemetry.py
	cd training && ../$(PY) train.py --out ../models

build:
	cmake -S . -B $(BUILD) -G "Unix Makefiles"
	cmake --build $(BUILD) -j$(JOBS)

test: model build test-python
	ctest --test-dir $(BUILD) --output-on-failure

test-python:
	cd training && ../$(PY) -m pytest -q ../tests/python

# Sanitizer builds exercise the pipeline, HTTP server and queue under TSan / ASan+UBSan.
test-tsan: model
	cmake -S . -B build-tsan -DEDGESCORE_SANITIZE=thread -DEDGESCORE_BUILD_BENCH=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build-tsan -j$(JOBS) --target edgescore_tests
	TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1" ./build-tsan/edgescore_tests --gtest_filter=-Onnx.*

test-asan: model
	cmake -S . -B build-asan -DEDGESCORE_SANITIZE=address,undefined -DEDGESCORE_BUILD_BENCH=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build-asan -j$(JOBS) --target edgescore_tests
	ASAN_OPTIONS="detect_leaks=1" UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" ./build-asan/edgescore_tests

run: model build
	EDGESCORE_MODEL=models/model.onnx $(BUILD)/edgescore

bench: model build
	$(BUILD)/edgescore_bench

bench-compare: model build
	$(PY) benchmarks/compare.py

docker:
	docker build -t edgescore:latest .

kind:
	scripts/kind-up.sh

clean:
	rm -rf build build-tsan build-asan
