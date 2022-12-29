#ifndef INTERFACE_H
#define INTERFACE_H

#include <cstdio>
#include <cstdint>

struct TrainingData {
	float ** inputs;
	float ** outputs;
	int length;
};

class NeuralNetwork {
public:
	NeuralNetwork(FILE * file); // Tested - Passed
	NeuralNetwork(uint32_t layers,
				  uint32_t * neurons,
				  float learningRate,
				  uint32_t batchSize); // Tested - Passed
	~NeuralNetwork(); // Tested - Passed
	void setInputs(float * data); // Tested - Passed
	void feedforward(); // Tested - Passed
	void getOutputs(float * outputs); // Tested - Passed

	void setExpectedOutput(float * expectedOutput); // Tested - Passed
	void backpropagate(); // Tested - Passed
	void updateParameters(); // Tested - Passed
	void saveNetwork(FILE * file); // Tested - Passed

	// static void * trainerFunction(void * params);
	// static void * fileReaderFunction(void * params);

	void train(FILE * inputs, FILE * outputs); // UNTESTED
	void train(TrainingData data); // UNTESTED

	void randomizeNetwork(); // Tested - Passed

	void printNetwork(); // Tested - Passed

private:
	uint32_t layers;
	float learningRate;
	uint32_t batchSize;

	uint32_t totalNeuronCount;
	uint32_t totalWeightCount;

	uint32_t offsetToOutputs;

	uint32_t * neurons;
	uint32_t * weightCounts;
	uint32_t * neuronOffsets;
	uint32_t * weightOffsets;
	float * deltaValues;
	float * deltaWeights;
	float * neuronValues;
	float * weights;
	float * biases;

	float * expectedOutput_d;
	float * neuronValues_d;
	float * weights_d;
	float * biases_d;
	float * batchDeltaValues_d;
	float * deltaValues_d;
	float * deltaWeights_d;
	float * batchDeltaWeights_d;

	void doAllocation(); // Tested - Passed
	void loadNetwork(FILE * file); // Tested - Passed
	void copyFromGpu(); // Tested - Passed
	void copyToGpu(); // Tested - Passed
};

#endif