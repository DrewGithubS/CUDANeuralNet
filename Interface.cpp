#include <cstdlib>
#include <pthread.h>
#include <string.h>

#include "GPUFunctions.cudah"
#include "Interface.h"

NeuralNetwork::NeuralNetwork(FILE * file) {
	loadNetwork(file);
}

NeuralNetwork::NeuralNetwork(
							uint32_t layersIn,
							uint32_t * neuronsIn,
							float learningRateIn,
							uint32_t batchSizeIn) {

	layers = layersIn;
	neurons = (uint32_t *) malloc(layers * sizeof(uint32_t));
	memcpy(neurons, neuronsIn, layers * sizeof(uint32_t));
	learningRate = learningRateIn;
	batchSize = batchSizeIn;

	doAllocation();
}

NeuralNetwork::~NeuralNetwork() {
	free(neurons);
	free(deltaValues);
	free(deltaWeights);
	free(weightCounts);
	free(neuronOffsets);
	free(weightOffsets);
	free(weights);
	free(biases);
	free(neuronValues);

	gpuFree(neuronValues_d);
	gpuFree(weights_d);
	gpuFree(biases_d);
	gpuFree(deltaValues_d);
	gpuFree(deltaWeights_d);
	gpuFree(expectedOutput_d);
	gpuFree(batchDeltaValues_d);
	gpuFree(batchDeltaWeights_d);
}

void NeuralNetwork::doAllocation() {
	totalNeuronCount = 0;
	totalWeightCount = 0;
	offsetToOutputs  = 0;

	weightCounts  = (uint32_t *) malloc((layers-1) * sizeof(uint32_t));
	neuronOffsets = (uint32_t *) malloc(layers * sizeof(uint32_t));
	weightOffsets = (uint32_t *) malloc(layers * sizeof(uint32_t));

	for(int i = 0; i < layers; i++) {
		neuronOffsets[i] = totalNeuronCount;
		weightOffsets[i] = totalWeightCount;
		totalNeuronCount += neurons[i];
		weightCounts[i] = neurons[i] * neurons[i + 1];
		if(i < (layers - 1)) {
			totalWeightCount += neurons[i] * neurons[i + 1];
			offsetToOutputs += neurons[i];
		}
	}


	 
	deltaValues = (float *) malloc(totalNeuronCount * sizeof(float));
	deltaWeights = (float *) malloc(totalWeightCount * sizeof(float));
	weights = (float *) malloc(totalWeightCount * sizeof(float));
	biases  = (float *) malloc(totalNeuronCount * sizeof(float));
	neuronValues = (float *) malloc(totalNeuronCount * sizeof(float));

	gpuMalloc((void**) &deltaValues_d,  totalNeuronCount * sizeof(float));
	gpuMalloc((void**) &batchDeltaValues_d, totalNeuronCount * sizeof(float));
	gpuMalloc((void**) &neuronValues_d, totalNeuronCount * sizeof(float));
	gpuMalloc((void**) &biases_d,       totalNeuronCount * sizeof(float));

	gpuMalloc((void**) &deltaWeights_d, totalWeightCount * sizeof(float));
	gpuMalloc((void**) &batchDeltaWeights_d, totalWeightCount * sizeof(float));
	gpuMalloc((void**) &weights_d,      totalWeightCount * sizeof(float));
	gpuMalloc((void**) &expectedOutput_d, neurons[layers - 1] * sizeof(float));
}

void NeuralNetwork::loadNetwork(FILE * file) {
	fread(&layers, 1, sizeof(uint32_t), file);
	fread(&learningRate, 1, sizeof(float), file);
	fread(&batchSize, 1, sizeof(uint32_t), file);
	neurons = (uint32_t *) malloc(layers * sizeof(uint32_t));
	fread(neurons, layers, sizeof(uint32_t), file);

	doAllocation();

	fread(biases,  totalNeuronCount, sizeof(float), file);
	fread(weights, totalWeightCount, sizeof(float), file);

	copyToGpu();
}

void NeuralNetwork::randomizeNetwork() {
	for(int i = 0; i < totalNeuronCount; i++) {
		biases[i] = (((float) (rand()) / (float) (RAND_MAX)) * 2) - 1;
	}

	for(int i = 0; i < totalWeightCount; i++) {
		weights[i] = (((float) (rand()) / (float) (RAND_MAX)) * 2) - 1;
	}

	copyToGpu();
}


void NeuralNetwork::copyToGpu() {
	cpuToGpuMemcpy(neuronValues_d, neuronValues, 
		totalNeuronCount * sizeof(float));
	cpuToGpuMemcpy(biases_d,  biases,  totalNeuronCount * sizeof(float));
	cpuToGpuMemcpy(weights_d, weights, totalWeightCount * sizeof(float));
}

void NeuralNetwork::copyFromGpu() {
	gpuToCpuMemcpy(deltaValues, deltaValues_d, 
		totalNeuronCount * sizeof(float));
	gpuToCpuMemcpy(neuronValues, neuronValues_d, 
		totalNeuronCount * sizeof(float));
	gpuToCpuMemcpy(biases,  biases_d,  totalNeuronCount * sizeof(float));
	gpuToCpuMemcpy(weights, weights_d, totalWeightCount * sizeof(float));
	gpuToCpuMemcpy(deltaWeights, deltaWeights_d, totalWeightCount * sizeof(float));
}

void NeuralNetwork::saveNetwork(FILE * file) {
	fwrite(&layers, 1, sizeof(uint32_t), file);
	fwrite(&learningRate, 1, sizeof(float), file);
	fwrite(&batchSize, 1, sizeof(uint32_t), file);
	fwrite(neurons, layers, sizeof(uint32_t), file);

	copyFromGpu();

	fwrite(biases,  totalNeuronCount, sizeof(float), file);
	fwrite(weights, totalWeightCount, sizeof(float), file);
}

void NeuralNetwork::setInputs(float * data) {
	cpuToGpuMemcpy(neuronValues_d, data, neurons[0] * sizeof(float));
}

void NeuralNetwork::getOutputs(float * outputs) {
	gpuToCpuMemcpy(neuronValues, 
				   neuronValues_d,
				   totalNeuronCount * sizeof(float));
	memcpy(outputs,
		&(neuronValues[neuronOffsets[layers-1]]),
		neurons[layers - 1] * sizeof(float));
}

void NeuralNetwork::setExpectedOutput(float * expectedOutput) {
	cpuToGpuMemcpy(expectedOutput_d,
				   expectedOutput,
				   neurons[layers - 1] * sizeof(float));
}

void NeuralNetwork::feedforward() {
	gpuFeedforward(
		layers,
		neurons,
		neuronOffsets,
		weightOffsets,
		neuronValues_d,
		biases_d,
		weights_d);

}

void NeuralNetwork::backpropagate() {
	gpuBackpropagate(
		layers,
		neurons,
		weightCounts,
		neuronOffsets,
		weightOffsets,
		expectedOutput_d,
		neuronValues_d,
		weights_d,
		deltaValues_d,
		batchDeltaValues_d,
		deltaWeights_d,
		batchDeltaWeights_d,
		totalWeightCount);
}

void NeuralNetwork::updateParameters() {
	gpuUpdateParameters(deltaValues_d,
						deltaWeights_d,
						weights_d,
						biases_d,
						totalWeightCount,
						totalNeuronCount,
						learningRate);
}

struct FileReaderParam {
	// Shared data

	// 0 is reading from file
	// 1 is training on network
	bool * dataState;
	pthread_mutex_t * dataStateMutex;
	bool * continueCheck;
	pthread_mutex_t * continueCheckMutex;
	float ** inputs;
	float ** expectedOutputs;
	
	// Task-specific data
	FILE * inputFile;
	FILE * outputFile;
	uint32_t inputLength;
	uint32_t outputLength;
};

struct TrainerParam {
	// Shared data

	// 0 is reading from file
	// 1 is training on network
	bool * dataState;
	pthread_mutex_t * dataStateMutex;
	bool * continueCheck;
	pthread_mutex_t * continueCheckMutex;
	float ** inputs;
	float ** expectedOutputs;

	// Task-specific data
	uint32_t totalNeuronCount;
	uint32_t batchSize;
	float * deltaValues_d;
	float * deltaWeights_d;
	NeuralNetwork * object;
};

// Used for allocating 2 different sets of data
// so that the reader thread and trainer thread
// alternate using each set.
#define ALTERNATOR_SIZE (2)

void * fileReaderFunction(void * params) {
	FileReaderParam * fParam = (FileReaderParam *) params;
	for(;;) {
		for(int i = 0; i < ALTERNATOR_SIZE; i++) {
			pthread_mutex_lock(&(fParam->dataStateMutex[i]));
			if(!fParam->dataState[i]) {
				if(!feof(fParam->inputFile) && !feof(fParam->outputFile)) {
					fread(fParam->inputs[i],
						  fParam->inputLength,
						  sizeof(float),
						  fParam->inputFile);

					fread(fParam->expectedOutputs[i],
						  fParam->outputLength,
						  sizeof(float),
						  fParam->outputFile);
				}
			}
			pthread_mutex_unlock(&(fParam->dataStateMutex[i]));
		}

		// The files should be aligned so both feof at the same time
		if(feof(fParam->inputFile) || feof(fParam->outputFile)) {
			pthread_mutex_lock(fParam->continueCheckMutex);
			*(fParam->continueCheck) = false;
			pthread_mutex_unlock(fParam->continueCheckMutex);
			break;
		}
	}
}

void * trainerFunction(void * params) {
	TrainerParam * tParam = (TrainerParam *) params;

	int iterationCount = 0;

	gpuMemset(
		tParam->deltaValues_d,
		0,
		tParam->totalNeuronCount * sizeof(float));

	gpuMemset(
		tParam->deltaWeights_d,
		0,
		tParam->totalNeuronCount * sizeof(float));

	for(;;) {
		for(int i = 0; i < ALTERNATOR_SIZE; i++) {
			pthread_mutex_lock(&(tParam->dataStateMutex[i]));
			if(tParam->dataState[i]) {
				tParam->object->setInputs(tParam->inputs[i]);
				tParam->object->feedforward();
				tParam->object->setExpectedOutput(tParam->expectedOutputs[i]);
				tParam->object->backpropagate();
			}
			
			pthread_mutex_unlock(&(tParam->dataStateMutex[i]));

			if(iterationCount == tParam->batchSize) {
				tParam->object->updateParameters();
				iterationCount = 0;
				gpuMemset(
					tParam->deltaValues_d,
					0,
					tParam->totalNeuronCount * sizeof(float));

				gpuMemset(
					tParam->deltaWeights_d,
					0,
					tParam->totalNeuronCount * sizeof(float));
			}
		}

		pthread_mutex_lock(tParam->continueCheckMutex);
		if(*(tParam->continueCheck) == false) {
			break;
		}
		pthread_mutex_unlock(tParam->continueCheckMutex);
	}
}

void NeuralNetwork::trainChunks(FILE * inputs, FILE * outputs) {
	bool * dataState = (bool *) malloc(ALTERNATOR_SIZE * sizeof(bool));

	pthread_mutex_t dataStateMutex[ALTERNATOR_SIZE];

	bool continueCheck = true;

	pthread_mutex_t continueCheckMutex;
	pthread_mutex_init(&continueCheckMutex, NULL);

	float ** inputData =
		(float **) malloc(ALTERNATOR_SIZE * sizeof(float *));
	float ** expectedOutputs =
		(float **) malloc(ALTERNATOR_SIZE * sizeof(float *));

	for(int i = 0; i < ALTERNATOR_SIZE; i++) {
		dataState[i] = false;
		inputData[i] = (float *) malloc(neurons[0] * sizeof(float));
		expectedOutputs[i] =
			(float *) malloc(neurons[layers - 1] * sizeof(float));
		pthread_mutex_init(&(dataStateMutex[i]), NULL);
	}

	// TODO: CONTINUE ASSIGNING STRUCTS
	FileReaderParam fileParam = (FileReaderParam) {
		.dataState = dataState,
		.dataStateMutex = dataStateMutex,
		.continueCheck = &continueCheck,
		.continueCheckMutex = &continueCheckMutex,
		.inputs = inputData,
		.expectedOutputs = expectedOutputs,
		.inputFile = inputs,
		.outputFile = outputs,
		.inputLength = neurons[0],
		.outputLength = neurons[layers - 1],
	};

	TrainerParam trainerParam = (TrainerParam) {
		.dataState = dataState,
		.dataStateMutex = dataStateMutex,
		.continueCheck = &continueCheck,
		.continueCheckMutex = &continueCheckMutex,
		.inputs = inputData,
		.expectedOutputs = expectedOutputs,
		.totalNeuronCount = totalNeuronCount,
		.batchSize = batchSize,
		.deltaValues_d = deltaValues_d,
		.deltaWeights_d = deltaWeights_d,
		.object = this,
	};

	pthread_t fileThread;
	pthread_t trainerThread;
	pthread_create(&fileThread, NULL, &fileReaderFunction, NULL);
	pthread_create(&trainerThread, NULL, &trainerFunction, NULL);
	pthread_join(fileThread, NULL);
    pthread_join(trainerThread, NULL);
}

void NeuralNetwork::trainChunks(TrainingData data) {
	for(int i = 0; i < data.length; i += batchSize) {
		if(i + batchSize < data.length) {
			// NOTE: If the float standard changes, this code breaks...
			gpuMemset(deltaValues_d, 0, totalNeuronCount * sizeof(float));
			gpuMemset(deltaWeights_d, 0, totalNeuronCount * sizeof(float));

			for(int j = 0; j < batchSize; j++) {
				setInputs(data.inputs[i]);
				feedforward();
				setExpectedOutput(data.outputs[i]);
				backpropagate();
			}
			updateParameters();
		}
		if(i % 100 == 0) {
			printf("%d\n", i);
		}
	}
}

void NeuralNetwork::trainBarrage(TrainingData data) {
	float * valueList;
	float * expectedOutputList;
	gpuMalloc((void**) &valueList,
		data.length * neurons[0] * sizeof(float));
	gpuMalloc((void**) &expectedOutputList,
		data.length * neurons[layers - 1] * sizeof(float));

	cpuToGpuMemcpy(valueList,
		data.inputs,
		data.length * neurons[0] * sizeof(float));

	cpuToGpuMemcpy(expectedOutputList,
		data.outputs,
		data.length * neurons[layers - 1] * sizeof(float));

	for(int i = 0; i < data.length; i += batchSize) {
		if(i + batchSize < data.length) {
			// NOTE: If the float standard changes, this code breaks...
			gpuMemset(deltaValues_d, 0, totalNeuronCount * sizeof(float));
			gpuMemset(deltaWeights_d, 0, totalNeuronCount * sizeof(float));

			for(int j = 0; j < batchSize; j++) {
				gpuToGpuMemcpy(neuronValues_d,
					valueList + i * batchSize + j,
					data.length * neurons[0] * sizeof(float));

				feedforward();
				gpuToGpuMemcpy(expectedOutput_d,
				   expectedOutputList + + i * batchSize + j,
				   neurons[layers - 1] * sizeof(float));
				backpropagate();
			}
			updateParameters();
		}
		if(i % 100 == 0) {
			printf("%d\n", i);
		}
	}
}

void NeuralNetwork::printNetwork() {
	copyFromGpu();
	for(int layer = 0; layer < layers; layer++) {
		printf("Layer %d:\n", layer);
		for(int neuron = 0; neuron < neurons[layer]; neuron++) {
			printf("\tNeuron %d:\n", neuron);
			printf("\t\tValue: %f\n", neuronValues[neuronOffsets[layer] + neuron]);
			printf("\t\tDelta Value: %f\n", deltaValues[neuronOffsets[layer] + neuron]);
			if(layer != 0) {
				printf("\t\tBias: %f\n", biases[neuronOffsets[layer] + neuron]);
			}
			if(layer < (layers - 1)) {
				printf("\t\tWeights:\n");
				for(int weight = 0; weight < neurons[layer + 1]; weight ++) {
					printf("\t\t\tWeight %d: %f\n", weight, 
						weights[weightOffsets[layer] + weight + 
						neurons[layer + 1] * neuron]);
				}
				printf("\t\tDelta Weights:\n");
				for(int weight = 0; weight < neurons[layer + 1]; weight ++) {
					printf("\t\t\tDelta Weight %d%d%d: %f\n",
						layer,
						neuron,
						weight, 
						deltaWeights[weightOffsets[layer] + weight + 
						neurons[layer + 1] * neuron]);
				}
			}
		}
	}
}