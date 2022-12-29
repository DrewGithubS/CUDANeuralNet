#include <cstdio>

#include "Interface.h"

int main() {
	FILE * file = fopen("Network.net", "rb");

	uint32_t neurons[] = {2, 3, 1};
	// NeuralNetwork network = NeuralNetwork(3, neurons, 0.01, 5);
	// network.randomizeNetwork();

	NeuralNetwork network = NeuralNetwork(file);
	// network.saveNetwork(file);

	float input[2] = {1, 1};
	float expectedOutput[1] = {0.1};
	float output[1];


	network.setInputs(input);
	network.feedforward();
	network.getOutputs(output);
	network.setExpectedOutput(expectedOutput);
	network.backpropagate();
	network.updateParameters();
	network.printNetwork();
	printf("Output: %lf\n", output[0]);
}