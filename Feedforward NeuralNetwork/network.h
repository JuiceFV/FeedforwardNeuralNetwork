#pragma once
#include "neuron.h"
#include <vector>

class network
{
public:
	network(const std::vector<unsigned> &topology);
	void feed_forward(const std::vector<double> &input_values);
	void back_propagation(const std::vector<double> &target_values);
	void get_result(std::vector<double> &result_values) const;
	double get_recent_averages() const;
private:
	std::vector<layer> layers;



};

