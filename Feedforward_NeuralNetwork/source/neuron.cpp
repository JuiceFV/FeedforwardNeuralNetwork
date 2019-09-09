#include "neuron.h"
#include <cmath>

double neuron::eta = 0.11;
double neuron::alpha = 0.5;

void neuron::set_output_value(double value) { output_value = value; }
double neuron::get_output_value() const { return (output_value); }
double neuron::random_weight() { return (rand() / double (RAND_MAX)); }
neuron::neuron(unsigned number_of_outputs, unsigned my_index)
{
	//creating connections with other neurons while
	//we creating a neuron
	for (unsigned i = 0; i < number_of_outputs; i++)
	{
		output_weights.push_back(connection());
		//set weight with random value in range [0;1]
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
double neuron::sumDOW(const layer &next_layer) const
{
	double sum = 0.0;
	for (unsigned i = 0; i < next_layer.size() - 1; i++)
	{
		sum += output_weights[i].weight * next_layer[i].gradient;
	}
	return (sum);
}

void neuron::calculate_hidden_gradients(const layer &next_layer)
{
	double dow = sumDOW(next_layer);
	gradient = dow * neuron::activation_function_derivative(output_value);
}

void neuron::calculate_output_gradients(double target_value)
{
	double delta = target_value - output_value;
	gradient = delta * neuron::activation_function_derivative(output_value);
}

double neuron::activation_function(double x)
{
	/* tanges
	return (tanh(x));
	*/
	

	/* SoftExponential
	if (alpha < 0)
		return (-1 * (log(1 - alpha * (x + alpha)) / alpha));
	else if (alpha == 0)
		return (x);
	else
		return (((exp(alpha * x) - 1) / alpha) + alpha);
		*/
	if (x <= 0)
		return (alpha * (exp(x) - 1));
	else
		return (x);
}

double neuron::activation_function_derivative(double x)
{
	/* tanges - deriv
	return (1 - pow(x, 2));
	*/
	/* SoftExponential - deriv
	if (alpha <  0)
		return (1/(1 - alpha * (alpha + x)));
	else
		return (exp(alpha * x));
		*/
	if (x <= 0)
		return (activation_function(x) + alpha);
	else
		return (1);
}

void neuron::feed_forward(const layer &previous_layer)
{
	double sum = 0.0;
	for (unsigned i = 0; i < previous_layer.size(); i++)
	{
		sum += previous_layer[i].get_output_value() *
				previous_layer[i].output_weights[this->my_index].weight;
	}
	output_value = neuron::activation_function(sum);
}


