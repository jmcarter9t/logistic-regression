target_sources(lr PRIVATE "${CMAKE_SOURCE_DIR}/cpp_utilities/src/utilities.cpp")

target_sources(lr PRIVATE "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
target_sources(lr PRIVATE "${CMAKE_CURRENT_LIST_DIR}/lr.cpp")
target_sources(lr PRIVATE "${CMAKE_CURRENT_LIST_DIR}/confusion_matrix.cpp")
