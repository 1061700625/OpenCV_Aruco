cmake_minimum_required(VERSION 3.22)
project(demo)
find_package(OpenCV 4 REQUIRED)
link_libraries(${OpenCV_LIBS})

IF(UNIX)
    SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
ENDIF(UNIX)


set(CMAKE_CXX_STANDARD 20)
aux_source_directory(./aamed SOURCE_AAMED)
aux_source_directory(./ED_Lib SOURCE_ED)


add_executable(demo
        main.cpp ellipse.cpp ellipse.h
        ${SOURCE_AAMED} ${SOURCE_ED} ED_Lib/ArucoDetect.cpp ED_Lib/ArucoDetect.h)
include_directories(./aamed ./ED_Lib)
target_link_libraries(demo ${OpenCV_LIBS})
