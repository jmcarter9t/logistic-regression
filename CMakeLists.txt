cmake_minimum_required (VERSION 3.1)

project (lr LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3")

find_package( Threads )

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/spdlog/include")
include_directories("${CMAKE_SOURCE_DIR}/eigen-git-mirror/Eigen")
include_directories("${CMAKE_SOURCE_DIR}/cpp_utilities/include")

add_executable(lr "")
target_link_libraries(lr ${CMAKE_THREAD_LIBS_INIT})
include("src/CMakeLists.txt")
