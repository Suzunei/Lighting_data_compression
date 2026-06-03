cmake_minimum_required(VERSION 3.16)
project(HINEGBD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- 静态库 ----------
set(LIB_SOURCES
    HINEGBD.cpp
)

set(HEADERS
    HINEGBD.h
)

add_library(HINEGBD_Lib STATIC ${LIB_SOURCES} ${HEADERS})

target_include_directories(HINEGBD_Lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include
)

target_link_libraries(HINEGBD_Lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/UCommon_Runtime.lib
)

# ---------- 可执行文件 (Demo) ----------
add_executable(HINEGBD_Demo main.cpp)
target_link_libraries(HINEGBD_Demo PRIVATE HINEGBD_Lib)

# ---------- MSVC 编译选项 ----------
if(MSVC)
    target_compile_options(HINEGBD_Lib PRIVATE /wd4251 /utf-8)
    target_compile_options(HINEGBD_Demo PRIVATE /wd4251 /utf-8)
    set_target_properties(HINEGBD_Lib PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
    )
    set_target_properties(HINEGBD_Demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    )
endif()
