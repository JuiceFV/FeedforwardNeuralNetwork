#include "neuron.h"
#include <cmath>

void neuron::set_output_value(double value) { output_value = value; }
double neuron::get_output_value() const { return (output_value); }
double neuron::random_weight() { return (rand() / double (RAND_MAX)); }
neuron::neuron(unsigned number_of_outputs, unsigned my_index)
{
	for (unsigned i = 0; i < number_of_outputs; i++)
	{
		output_weights.push_back(connection());
		output_weights.back().weight = random_weight();
	}
	this->my_index = my_index;
}

void neuron::update_input_weights(layer &previous_layer)
{
	for (unsigned i = 0; i < previous_layer.size(); i++)
	{
			neuron &neuron = previous_layer[i];
			double old_delta_weight = neuron.output_weights[this->my_index].delta_weight;
			double new_delta_weight = eta * neuron.get_output_value() * gradient + alpha * old_delta_weight;
			neuron.output_weights[this->my_index].delta_weight = new_delta_weight;
			neuron.output_weights[this->my_index].weight += new_delta_weight;
	}
}

