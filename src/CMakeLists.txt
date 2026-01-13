set(CXX_FILES "main.cpp" "TinyMD.cpp" "Atom.cpp")

add_executable(${PROJECT_NAME} ${CXX_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


include(FetchContent)

# TinyLA - disable tests and examples to avoid dependency issues
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(TINYLA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYLA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYLA_USE_CUDA OFF CACHE BOOL "Disable CUDA support in TinyLA" FORCE)

FetchContent_Declare(TinyLA_content
    GIT_REPOSITORY https://github.com/TheFlyingPiano99/TinyLA.git
    GIT_TAG main
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(TinyLA_content)
target_link_libraries(${PROJECT_NAME} PRIVATE TinyLA::TinyLA)
