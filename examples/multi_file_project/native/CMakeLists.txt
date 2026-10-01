cmake_minimum_required(VERSION 3.16)
project(easymodel_native_example LANGUAGES CXX)

add_library(easymodel_native SHARED math.cpp)
target_compile_features(easymodel_native PRIVATE cxx_std_17)
