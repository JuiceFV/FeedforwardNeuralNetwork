#include "training_set.h"
#include "network.h"

double network::recent_average_smoothing_factor = 100.0;

network::network(const std::vector<unsigned> &topology)
{
	//get the number of layers from the size of
	//the topology, we will insert each layer into our layers (NN)
	unsigned number_of_layers = topology.size();
	for (unsigned layer_num = 0; layer_num < number_of_layers; layer_num++)
	{
		//inserting layers
		layers.push_back(layer());
		//if current layer is last layer number of our outputs (connections) is zero
		//otherwise (current layer isn't last) number of outputs (connections) are
		//equal to the number of neurons at the next layer
		unsigned number_of_outputs = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];
		//inserting neurons in an each layer
		for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; neuron_num++)
		{
			//passing numbers of outputs while we creating each neuron and pushing it to each layer
			layers.back().push_back(neuron(number_of_outputs, neuron_num));
		}
		//its Bayes neuron (bayes shift value I mean)
		//This value here is for curve-adjustment
		layers.back().back().set_output_value(1.0);
	}
}

void network::get_result(std::vector<double> &result_values) const
{
	result_values.clear();

	for (unsigned i = 0; i < layers.back().size() - 1; i++)
	{
		result_values.push_back(layers.back()[i].get_output_value());
	}
}

//A little bit of mathematics (statistics I suppose) :)
//I chose the first metric that I recognized (it was first
//metric that I learned)
//So...
//1) https://en.wikipedia.org/wiki/Root-mean-square_deviation (RMSE)
//2) https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4 (how to choose a metric)
//3) https://en.wikipedia.org/wiki/Gradient (gradient)
void network::back_propagation(const std::vector<double> &target_values)
{
	layer &output_layer = layers.back();
	error = 0.0;
	//*----------------------RMSE-----------------------------------------------*//
	for (unsigned i = 0; i < output_layer.size() - 1; i++)
	{
		double delta = target_values[i] - output_layer[i].get_output_value();
		error += delta * delta;
	}
	error /= output_layer.size() - 1;
	error = sqrt(error);
	//*----------------------END-OF-RMSE-----------------------------------------------*//

	recent_average_error =
			(recent_average_error * recent_average_smoothing_factor + error)
			/ (recent_average_smoothing_factor + 1.0);

	//calculate a gradient for output layer
	for (unsigned i = 0; i < output_layer.size() - 1; i++)
	{
		output_layer[i].calculate_output_gradients(target_values[i]);
	}

	//calculate a gradient for hidden layer
	for (unsigned layer_num = layers.size() - 2; layer_num > 0; layer_num--)
	{
		layer &hidden_layer = layers[layer_num];
		layer &next_layer = layers[layer_num + 1];

		for (unsigned i = 0; i < hidden_layer.size(); i++)
		{
			hidden_layer[i].calculate_hidden_gradients(next_layer);
		}
	}

	//So we need to update weights using conclusions that we got before
	for (unsigned layer_num = layers.size() - 1; layer_num > 0; layer_num--)
	{
		layer &_layer = layers[layer_num];
		layer &previous_layer = layers[layer_num - 1];

		for (unsigned i = 0; i < _layer.size() - 1; i++)
		{
			_layer[i].update_input_weights(previous_layer);
		}
	}
}

//method that feeds a network with input values
//note: neuron's feed-forward method is different by
//		network's feed-forward method
void network::feed_forward(const std::vector<double> &input_values)
{
	assert(input_values.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < input_values.size(); i++)
	{
		//here we set the input for each neuron.
		//Because of this is only input, then we set 
		//these values ​​only for the first layer
		layers[0][i].set_output_value(input_values[i]);
	}

	for (unsigned layer_num = 1; layer_num < layers.size(); layer_num++)
	{
		layer &prev_layer = layers[layer_num - 1];
		for (unsigned i = 0; i < layers[layer_num].size() - 1; i++)
		{
			//here each (but not first layer) gets their
			//input values by summing up outputs values
			//of previous layer and passing up through a 
			//nonlinear function (Exponential linear unit)
			layers[layer_num][i].feed_forward(prev_layer);
		}
	}
}

double network::get_recent_average_error() const
{
	return (recent_average_error);
}