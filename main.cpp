#include <cstdio>
#include <stdlib.h>

#include "Interface.h"

#define IMAGES (60000)
#define ROWS (28)
#define COLUMNS (28)
#define LABEL_COUNT (10)

#define MAX(x, y) (x > y ? x : y)

int main() {

	uint32_t neurons[] = {ROWS * COLUMNS, 500, 500, 10};
	NeuralNetwork * network = new NeuralNetwork(4, neurons, 0.001, 5);
	network->randomizeNetwork();

	// NeuralNetwork * network = new NeuralNetwork(file);

	FILE * imageData = fopen("data/imageData.bin", "rb");
	FILE * labelData = fopen("data/labelData.bin", "rb");


	float ** inputs = (float **) malloc(IMAGES * sizeof(float *));
	float ** expectedOutputs = (float **) malloc(IMAGES * sizeof(float *));
	for(int i = 0; i < IMAGES; i++) {
		inputs[i] = (float *) malloc(ROWS * COLUMNS * sizeof(float));
		expectedOutputs[i] = (float *) malloc(LABEL_COUNT * sizeof(float));
		fread(inputs[i], ROWS * COLUMNS, sizeof(float), imageData);
		fread(expectedOutputs[i], LABEL_COUNT, sizeof(float), labelData);
	}

	TrainingData trainer;
	trainer.inputs = inputs;
	trainer.outputs = expectedOutputs;
	trainer.length = IMAGES;
	network->trainChunks(trainer);
	// float * loss = (float *) malloc(IMAGES * sizeof(float));
	// float * output = (float *) malloc(LABEL_COUNT * sizeof(float));
	// for(int i = 0; i < IMAGES; i++) {
	// 	network->setInputs(inputs[i]);
	// 	network->feedforward();
	// 	network->getOutputs(output);
	// 	loss[i] = 0;
	// 	for(int j = 0; j < LABEL_COUNT; j++) {
	// 		float error = output[j] - expectedOutputs[i][j];
	// 		loss[i] += MAX(error, -error);
	// 	}
	// 	network->setExpectedOutput(expectedOutputs[i]);
	// 	network->backpropagate();
	// 	if(i % 5 == 0) {
	// 		network->updateParameters();
	// 	}
	// 	if(i % 100 == 0) {
	// 		printf("%d\n", i);
	// 	}
	// }
	// network->printNetwork();
	// network->getOutputs(output);
	
	// FILE * csv = fopen("loss.csv", "w");

	// for(int i = 0; i < IMAGES; i++) {
		// fprintf(csv, "%f\n", loss[i]);
	// }

	FILE * file = fopen("MNISTNet.net", "wb");
	printf("Saving...\n"); fflush(stdout);
	network->saveNetwork(file);
	printf("Saving complete.\n"); fflush(stdout);

	delete network;
}

// Example program...
/*
int main() {
	FILE * file = fopen("Network.net", "rb");

	uint32_t neurons[] = {2, 3, 1};
	// NeuralNetwork network = NeuralNetwork(3, neurons, 0.01, 5);
	// network.randomizeNetwork();

	NeuralNetwork * network = new NeuralNetwork(file);
	// network.saveNetwork(file);

	float input[2] = {1, 1};
	float expectedOutput[1] = {0.1};
	float output[1];


	network->setInputs(input);
	network->feedforward();
	network->getOutputs(output);
	network->setExpectedOutput(expectedOutput);
	network->backpropagate();
	network->updateParameters();
	network->printNetwork();
	printf("Output: %lf\n", output[0]);

	float * loss = (float *) malloc(ITERATIONS * sizeof(float));

	for(int i = 0; i < ITERATIONS; i++) {
		network->setInputs(input);
		network->feedforward();
		network->getOutputs(output);
		loss[i] = output[0] - expectedOutput[0];
		network->setExpectedOutput(expectedOutput);
		network->backpropagate();
		network->updateParameters();
	}
	network->printNetwork();
	network->getOutputs(output);
	
	printf("Loss: ");
	for(int i = 0; i < ITERATIONS; i++) {
		printf("%f, ", loss[i]);
	}
	printf("\n");
	printf("Output: %f\n", output[0]);

	delete network;
}
*/