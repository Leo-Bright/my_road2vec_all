cmake_minimum_required(VERSION 3.13)
project(my_road2vec C)

set(CMAKE_C_STANDARD 99)

link_libraries(m pthread)

add_executable(my_road2vec main.c)