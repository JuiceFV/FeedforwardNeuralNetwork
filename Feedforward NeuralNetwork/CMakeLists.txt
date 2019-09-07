cmake_minimum_required(VERSION 3.14)
project(Feedforward_NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(Feedforward_NeuralNetwork main.cpp neuron.h neuron.cpp network.cpp network.h)