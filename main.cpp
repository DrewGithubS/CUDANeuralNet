#include "Interface.h"

int main() {
	uint32_t neurons[] = {2, 3, 1};
	NeuralNetwork network = NeuralNetwork(3, neurons, 0.01, 5);

	float input[2] = {1, 1};
	float expectedOutput[1] = {0.1};
	float output[1];


	network.setInputs(input);
	network.feedforward();
	network.getOutputs(output);
	printf("Output: %lf\n", output[0]);
	network.setExpectedOutput(expectedOutput);
	// network.backpropagate();
}