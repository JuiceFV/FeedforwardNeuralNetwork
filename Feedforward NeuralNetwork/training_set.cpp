#include "training_set.h"

training_set::training_set(const std::string file_name)
{
	training_data_file.open(file_name.c_str());
}

bool training_set::is_eof()
{
	return (training_data_file.eof());
}

void training_set::get_topology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	std::getline(training_data_file, line);
	std::stringstream os(line);

	os >> label;
	if (this->is_eof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!os.eof())
	{
		unsigned n;
		os >> n;
		topology.push_back(n);
	}
	return;
}

unsigned training_set::get_next_inputs(std::vector<double> &input_values)
{
	input_values.clear();

	std::string line;
	std::getline(training_data_file, line);
	std::stringstream os(line);

	std::string label;
	os >> label;
	if (label.compare("in:") == 0)
	{
		double one_value;
		while (os >> one_value)
		{
			input_values.push_back(one_value);
		}
	}
	return (input_values.size());
}

unsigned training_set::get_target_outputs(std::vector<double> &target_output_values)
{
	target_output_values.clear();

	std::string line;
	std::getline(training_data_file, line);
	std::stringstream os(line);

	std::string label;
	os >> label;
	if (label.compare("out:") == 0)
	{
		double one_value;
		while (os >> one_value)
		{
			target_output_values.push_back(one_value);
		}
	}
	return (target_output_values.size());
}