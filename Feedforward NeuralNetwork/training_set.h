#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class training_set
{
public:
	training_set(const std::string file_name);
	bool is_eof();
	void get_topology(std::vector<unsigned> &topology);
	unsigned get_next_inputs(std::vector<double> &input_values);
	unsigned get_target_outputs(std::vector<double> &target_output_values);
private:
	std::ifstream training_data_file;
};
