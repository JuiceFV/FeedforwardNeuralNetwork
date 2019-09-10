/*
For training I used Gate NaNd values which stored in testData.txt
*/
#include "training_set.h"
#include "network.h"
#include "neuron.h"

void show_vectors_values(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

int main()
{
	training_set training_data("testData.txt");
	std::vector<unsigned> topology;
	training_data.get_topology(topology);
	network net(topology);

	std::vector<double> input_values, target_values, result_values;
	int training_pass = 0;
	while (!training_data.is_eof())
	{
		training_pass++;
		std::cout << "Pass: " << training_pass << std::endl;

		if (training_data.get_next_inputs(input_values) != topology[0])
			break;
		show_vectors_values("Input:", input_values);
		net.feed_forward(input_values);

		training_data.get_target_outputs(target_values);
		show_vectors_values("Targets:", target_values);
		assert(target_values.size() == topology.back());

		net.get_result(result_values);
		show_vectors_values("Outputs:", result_values);

		net.back_propagation(target_values);

		std::cout << "Net average error: " << net.get_recent_average_error() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;
#if defined(_MSC_VER) || defined(_WIN32)
	system("pause");
#endif
	return (0);
}
