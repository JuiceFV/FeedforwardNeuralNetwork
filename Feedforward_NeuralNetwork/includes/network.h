#pragma once
#include "neuron.h"
#include <vector>

class network
{
public:
	//constructor in which we'ill pass topology (neural network structure)
	network(const std::vector<unsigned> &topology);
	//feed-forward - it is a method of filling a network
	void feed_forward(const std::vector<double> &input_values);
	//back propagation - learning algorithm
	void back_propagation(const std::vector<double> &target_values);
	//get results after an approximation
	void get_result(std::vector<double> &result_values) const;
	double get_recent_average_error() const;
private:
	//layers which contain array with neurons
	//it's like matrix-array (layers[layer][neuron])
	std::vector<layer> layers;
	double error;
	//its just helper for us 
	//its an average value of network error
	double recent_average_error;
	static double recent_average_smoothing_factor;
};

