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
	NeuralNetwork(FILE * file);
	NeuralNetwork(uint32_t layers,
				  uint32_t * neurons,
				  float learningRate,
				  uint32_t batchSize);
	~NeuralNetwork();
	void setInputs(float * data);
	void feedforward();
	void getOutputs(float * outputs);

	void setExpectedOutput(float * expectedOutput);
	void backpropagate();
	void updateParameters();
	void saveNetwork(FILE * file);

	// static void * trainerFunction(void * params);
	// static void * fileReaderFunction(void * params);

	void train(FILE * inputs, FILE * outputs);
	void train(TrainingData data);

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

	void doAllocation();
	void loadNetwork(FILE * file);
};

#endif