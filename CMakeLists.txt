cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(TinyMD LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
