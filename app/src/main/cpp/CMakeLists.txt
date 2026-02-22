cmake_minimum_required(VERSION 3.22)
project(docshot_native)

add_library(docshot_native SHARED
    directional_gradient.cpp
    jni_bridge.cpp
)

target_compile_options(docshot_native PRIVATE
    -O2 -ffast-math -Wall -Wextra
)

target_link_libraries(docshot_native
    log
)
