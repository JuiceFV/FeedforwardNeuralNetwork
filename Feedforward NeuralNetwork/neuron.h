#pragma once
#include <vector>

class neuron;

typedef std::vector<neuron> layer;

struct connection
{
	double weight;
	double delta_weight;
};

class neuron
{
public:
	neuron(unsigned number_of_outputs, unsigned my_index);
	void set_output_value(double value);
	double get_output_value() const;
	void feed_forward(const layer &previous_layer);
	void calculate_output_gradients(double target_value);
	void calculate_hidden_gradients(const layer &next_layer);
	void update_input_weights(layer &previous_layer);
private:
	static double eta;
	static double alpha;
	static double random_weight();
	static double activation_function(double x);
	static double activation_function_derivative(double x);
	double sumDOW(const layer &next_layer) const;
	double output_value;
	unsigned my_index;
	double gradient;
	std::vector<connection> output_weights;
};


