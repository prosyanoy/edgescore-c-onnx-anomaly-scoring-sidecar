cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
project(edgescore LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(EDGESCORE_BUILD_TESTS "Build GoogleTest suite" ON)
option(EDGESCORE_BUILD_BENCH "Build Google Benchmark micro-benchmarks" ON)
set(EDGESCORE_SANITIZE "" CACHE STRING "Sanitizer to enable: thread | address | undefined | address,undefined")
set(ORT_ROOT "" CACHE PATH "Pre-extracted ONNX Runtime release (include/ + lib/). Downloaded if empty.")
set(ORT_VERSION "1.22.0" CACHE STRING "ONNX Runtime release to download when ORT_ROOT is empty")

if(EDGESCORE_SANITIZE)
  add_compile_options(-fsanitize=${EDGESCORE_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${EDGESCORE_SANITIZE})
endif()

include(FetchContent)

# ---- ONNX Runtime (prebuilt CPU release) -------------------------------------------------
if(NOT ORT_ROOT)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ort_arch "aarch64")
  else()
    set(_ort_arch "x64")
  endif()
  FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${_ort_arch}-${ORT_VERSION}.tgz)
  FetchContent_MakeAvailable(onnxruntime)
  set(ORT_ROOT ${onnxruntime_SOURCE_DIR})
endif()
find_library(ORT_LIB onnxruntime PATHS ${ORT_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
add_library(onnxruntime_lib SHARED IMPORTED)
set_target_properties(onnxruntime_lib PROPERTIES
  IMPORTED_LOCATION ${ORT_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include)

# ---- Core library + service ---------------------------------------------------------------
find_package(Threads REQUIRED)
add_library(edgescore_core STATIC
  src/json.cpp src/metrics.cpp src/events.cpp src/model.cpp
  src/pipeline.cpp src/config.cpp src/http_server.cpp src/service.cpp)
target_include_directories(edgescore_core PUBLIC include)
target_link_libraries(edgescore_core PUBLIC onnxruntime_lib Threads::Threads)
target_compile_options(edgescore_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(edgescore src/main.cpp)
target_link_libraries(edgescore PRIVATE edgescore_core)
set_target_properties(edgescore PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib INSTALL_RPATH "$ORIGIN/../lib")

install(TARGETS edgescore RUNTIME DESTINATION bin)
install(DIRECTORY ${ORT_ROOT}/lib/ DESTINATION lib FILES_MATCHING PATTERN "libonnxruntime*.so*"
        PATTERN "cmake" EXCLUDE PATTERN "pkgconfig" EXCLUDE)

# ---- Tests --------------------------------------------------------------------------------
if(EDGESCORE_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_executable(edgescore_tests
    tests/test_json.cpp tests/test_queue.cpp tests/test_events.cpp tests/test_metrics.cpp
    tests/test_pipeline.cpp tests/test_config.cpp tests/test_http.cpp tests/test_onnx.cpp)
  target_link_libraries(edgescore_tests PRIVATE edgescore_core GTest::gtest_main)
  target_compile_definitions(edgescore_tests PRIVATE EDGESCORE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models")
  set_target_properties(edgescore_tests PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib)
  include(GoogleTest)
  gtest_discover_tests(edgescore_tests DISCOVERY_TIMEOUT 60)
endif()

# ---- Benchmarks ---------------------------------------------------------------------------
if(EDGESCORE_BUILD_BENCH)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_executable(edgescore_bench bench/bench_main.cpp)
  target_link_libraries(edgescore_bench PRIVATE edgescore_core benchmark::benchmark)
  target_compile_definitions(edgescore_bench PRIVATE EDGESCORE_MODEL_DIR="${CMAKE_SOURCE_DIR}/models")
  set_target_properties(edgescore_bench PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib)
endif()
