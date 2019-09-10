#include "neuron.h"
#include <cmath>

//network learning rate
//https://en.wikipedia.org/wiki/Learning_rate
double neuron::eta = 0.11;
//momentum
//https://medium.com/@abhinav.mahapatra10/ml-advanced-momentum-in-machine-learning-what-is-nesterov-momentum-ad37ce1935fc
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

//just update weights...
//explore the links above
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
double neuron::sum_derivative_of_weights(const layer &next_layer) const
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
	//because we don't have any output value, we will calculate 
	//a derivative of weights (summ of the mult of a weight and gradient
	// of each neuron)
	double deriv_of_weights = sum_derivative_of_weights(next_layer);
	//yeah we have gradient 
	gradient = deriv_of_weights * neuron::activation_function_derivative(output_value);
}

void neuron::calculate_output_gradients(double target_value)
{
	//we need to know our target value (the value that we would like to see)
	//and value which we got earlier
	double delta = target_value - output_value;
	//differentiate our output value and multiplying it to the delta which we got
	gradient = delta * neuron::activation_function_derivative(output_value);
	//Easy right? :P (It was really hard for me 7-8 month ago)
}

double neuron::activation_function(double x)
{
	//wiki will help you
	//https://en.wikipedia.org/wiki/Activation_function
	if (x <= 0)
		return (alpha * (exp(x) - 1.0));
	else
		return (x);
}

double neuron::activation_function_derivative(double x)
{
	//wiki will help you
	//https://en.wikipedia.org/wiki/Activation_function
	if (x <= 0)
		return (activation_function(x) + alpha);
	else
		return (1);
}

//feed-forward (neuron) function which we'ill use in
//feed-forward (NN) function for set input of each 
//neuron in each layer
void neuron::feed_forward(const layer &previous_layer)
{
	double sum = 0.0;
	for (unsigned i = 0; i < previous_layer.size(); i++)
	{
		//how I just said in "feed-forward (NN) function" description
		//here we just summing up multiplications of an output value 
		//with their weights ...
		sum += previous_layer[i].get_output_value() *
				previous_layer[i].output_weights[this->my_index].weight;
	}
	//... and using nonlinear function (ELU)
	output_value = neuron::activation_function(sum);
}


