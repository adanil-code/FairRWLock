#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default configuration values
BUILD_TYPE="Release"
COMPILER="g++"
CLEAN_BUILD=false
ENABLE_NUMA="OFF"

# Capture the exact directory where this script resides
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 0. Clean in-source cache if it exists (prevents configuration pollution)
if [ -f "$SCRIPT_DIR/CMakeCache.txt" ] || [ -d "$SCRIPT_DIR/CMakeFiles" ]; then
    echo "Cleaning in-source CMake cache to prevent build pollution..."
    rm -f "$SCRIPT_DIR/CMakeCache.txt"
    rm -rf "$SCRIPT_DIR/CMakeFiles"
fi

# 1. Parse Command Line Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--clean) 
            CLEAN_BUILD=true
            shift 
            ;;
        -t|--type) 
            BUILD_TYPE="$2"
            shift 2 
            ;;
        --compiler) 
            COMPILER="$2"
            shift 2 
            ;;
        --numa)
            ENABLE_NUMA="ON"
            shift
            ;;
        -h|--help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo "Options:"
            echo "  -c, --clean     Wipe build directory before starting"
            echo "  -t, --type      Build type (Release/Debug) [Default: Release]"
            echo "  --compiler      Compiler choice (g++, clang++) [Default: g++]"
            echo "  --numa          Enable NUMA-aware lock capabilities in the build"
            echo "  -h, --help      Display this help message"
            exit 0
            ;;
        *) 
            echo "Error: Unknown parameter: $1"
            exit 1 
            ;;
    esac
done

# 2. Wipe old build folder if a clean build was requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Wiping build directory..."
    rm -rf "$SCRIPT_DIR/build"
fi

echo "========================================"
echo " Building test_fair_rw_lock             "
echo " Build Type:   $BUILD_TYPE              "
echo " Compiler:     $COMPILER                "
echo " NUMA Support: $ENABLE_NUMA             "
echo "========================================"

# Create the build directory
mkdir -p "$SCRIPT_DIR/build"

# 3. Configure the project with CMake
echo "-> Configuring CMake..."
cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_COMPILER="$COMPILER" \
    -DENABLE_NUMA="$ENABLE_NUMA"

# 4. Compile the project
echo "-> Compiling with portable speed optimizations..."
# Automatically scales parallel compilation jobs (nproc on Linux, sysctl on macOS)
cmake --build "$SCRIPT_DIR/build" -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "========================================"
echo " Build successful!                      "
echo " Run via: ./build/bin/test_fair_rw_lock "
echo "========================================"
