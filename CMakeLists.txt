cmake_minimum_required(VERSION 3.20)
project(test_fair_rw_lock CXX)

# Force C++20 standard compliance for std::barrier, std::latch, etc.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce clean executable output directory structuring
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Portable cross-platform thread library support (Linux and macOS)
find_package(Threads REQUIRED)

# Link-Time Optimization (LTO/IPO)
# Optimizes across translation units for speed without targeting specific CPU instruction sets
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Configure compiler flags based on OS and choices (includes macOS AppleClang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # Base warning flags for all configurations
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    # - -O3: Maximum optimization (loop vectorization, aggressive inlining)
    # - -DNDEBUG: Disables standard C/C++ assertions for zero runtime overhead    
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
    )
endif()

# Define the testing binary target from the test/ directory
add_executable(test_fair_rw_lock
    test/fair_rw_lock_test.cpp
)

# Include both directories for headers in fair_rw_lock/ and test/
target_include_directories(test_fair_rw_lock PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fair_rw_lock
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

# Link platform-specific threading library
target_link_libraries(test_fair_rw_lock PRIVATE Threads::Threads)