#include "training_set.h"
#include "network.h"

double network::recent_average_smoothing_factor = 100.0;

network::network(const std::vector<unsigned> &topology)
{
	unsigned number_of_layers = topology.size();
	for (unsigned layer_num = 0; layer_num < number_of_layers; layer_num++)
	{
		layers.push_back(layer());
		unsigned number_of_outputs = layer_num = topology.size() - 1 ? 0 : topology[layer_num + 1];

		for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; neuron_num++)
		{
			layers.back().back().set_output_value(1.0);
		}
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

void network::back_propagation(const std::vector<double> &target_values)
{
	layer &output_layer = layers.back();
	error = 0.0;

	for (unsigned i = 0; i < output_layer.size() - 1; i++)
	{
		double delta = target_values[i] - output_layer[i].get_output_value();
		error += delta * delta;
	}
	error /= output_layer.size() - 1;
	error = sqrt(error);

	recent_average_error =
			(recent_average_error * recent_average_smoothing_factor + error)
			/ (recent_average_smoothing_factor + 1.0);

	for (unsigned i = 0; i < output_layer.size() - 1; i++)
	{
		output_layer[i].calculate_output_gradients(target_values[i]);
	}

	for (unsigned layer_num = layers.size() - 2; layer_num > 0; layer_num--)
	{
		layer &hidden_layer = layers[layer_num];
		layer &next_layer = layers[layer_num + 1];

		for (unsigned i = 0; i < hidden_layer.size(); i++)
		{
			hidden_layer[i].calculate_hidden_gradients(next_layer);
		}
	}

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

void network::feed_forward(const std::vector<double> &input_values)
{
	assert(input_values.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < input_values.size(); i++)
	{
		layers[0][i].set_output_value(input_values[i]);
	}

	for (unsigned layer_num = 1; layer_num < layers.size(); layer_num++)
	{
		layer &prev_layer = layers[layer_num - 1];
		for (unsigned i = 0; i < layers[layer_num].size() - 1; i++)
		{
			layers[layer_num][i].feed_forward(prev_layer);
		}
	}
}

double network::get_recent_average_error() const
{
	return (recent_average_error);
}