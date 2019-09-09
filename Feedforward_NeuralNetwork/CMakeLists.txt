cmake_minimum_required(VERSION 3.14)
project(Feedforward_NeuralNetwork)

set(Feedforward_NeuralNetwork_VERSION_MAJOR 1)
set(Feedforward_NeuralNetwork_VERSION_MINOR 0)
set(Feedforward_NeuralNetwork_VERSION ${Feedforward_NeuralNetwork_VERSION_MAJOR}.${Feedforward_NeuralNetwork_VERSION_MINOR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_EXC  ./source/main.cpp)
set(SOURCE ./source/neuron.cpp ./source/network.cpp ./source/training_set.cpp)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/testData.txt
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include_directories("${PROJECT_SOURCE_DIR}/includes")

add_executable(${PROJECT_NAME} ${SOURCE_EXC} ${SOURCE})
