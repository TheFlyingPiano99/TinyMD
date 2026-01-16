# Test configuration
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find or fetch Catch2
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    # Force Release build for Catch2
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Create test executable
add_executable(TinyMD_tests 
    test_velocity_verlet.cpp
    # test_rotation_matrix.cpp  # Temporarily disabled - needs refactoring
    test_rotational_motion.cpp
    test_coulomb_interaction.cpp
    ../src/TinyMD.cpp
)

# Link libraries
target_link_libraries(TinyMD_tests PRIVATE 
    Catch2::Catch2WithMain
    TinyLA::TinyLA
)

# Include directories
target_include_directories(TinyMD_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Enable testing
include(CTest)
include(Catch)
catch_discover_tests(TinyMD_tests)
