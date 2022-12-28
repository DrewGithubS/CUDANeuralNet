#include <cuda_runtime.h>

#include "GPUFunctions.cudah"

void gpuMalloc(void ** ptr, size_t size) {
	cudaMalloc(ptr, size);
}

void gpuFree(void * ptr) {
	cudaFree(ptr);
}

void cpuToGpuMemcpy(void * dest, void * src, size_t size) {
	cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void gpuToCpuMemcpy(void * dest, void * src, size_t size) {
	cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

void gpuMemset(void * ptr, int value, size_t count) {
	cudaMemset(ptr, value, count);
}

__global__ void simpleMatrixMult(
								float * neuronValues,
								float * weights,
								float * biases,
								uint32_t currentOffset,
								uint32_t nextOffset,
								uint32_t weightOffset,
								uint32_t currentNeurons,
								uint32_t nextNeurons) {

   uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < nextNeurons) {
		neuronValues[nextOffset + index] = biases[nextOffset + index];
    	for(uint32_t i = 0; i < nextNeurons; i++) {
    		neuronValues[nextOffset + index] += 
    			neuronValues[currentOffset + index] *
    			weights[weightOffset + i * nextNeurons + index];
    	}

    	// ReLU Activation
    	neuronValues[nextOffset + index] = 
    		neuronValues[nextOffset + index] > 0 ? 
    			neuronValues[nextOffset + index] :
    			-neuronValues[nextOffset + index];
	}
};

int threadsPerBlock = 1024;
void gpuFeedforward(
					int layers,
					uint32_t * neurons,
					uint32_t * neuronOffsets,
					uint32_t * weightOffsets,
					float * neuronValues,
					float * biases,
					float * weights) {

	int blocksPerGrid;
	for(int i = 0; i < layers-1; i++) {
	   	blocksPerGrid = 
	   		(neurons[i + 1] + threadsPerBlock - 1) / threadsPerBlock;

		simpleMatrixMult <<< blocksPerGrid, threadsPerBlock >>> (
			neuronValues,
			weights,
			biases,
			neuronOffsets[i],
			neuronOffsets[i + 1],
			weightOffsets[i],
			neurons[i],
			neurons[i + 1]);
	}
}

__global__ void setDeltaExpect(
								uint32_t neuronOffset,
								uint32_t neuronCount,
								float * neuronValues,
								float * expectedValues,
								float * deltaValues) {

   uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < neuronCount) {
		deltaValues[index] = 
			expectedValues[index] -
			neuronValues[neuronOffset + index];
	}
};

__global__ void getValueDeltas(
								float * batchDeltaValues,
								float * deltaValues,
								float * weights,
								uint32_t weightOffset,
								uint32_t currentOffset,
								uint32_t prevOffset,
								uint32_t currentNeurons,
								uint32_t prevNeurons) {

   uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < prevNeurons) {
		deltaValues[prevOffset + index] = 0;
		for(int i = 0; i < currentNeurons; i++) {
			deltaValues[prevOffset + index] += 
				weights[weightOffset + currentNeurons * index + i] *
				deltaValues[currentOffset + i];
		}
		batchDeltaValues[prevOffset + index] += 
			deltaValues[prevOffset + index];
	}
};

__global__ void getWeightDeltas(
								float * deltaValues,
								float * neuronValues,
								float * deltaWeights,
								float * batchDeltaWeights,
								uint32_t currentNeurons,
								uint32_t prevNeurons,
								uint32_t weightOffset,
								uint32_t prevOffset,
								uint32_t currOffset) {

   uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < prevNeurons * currentNeurons) {
		uint32_t prevNeuron = index / prevNeurons;
		uint32_t currNeuron = index - (prevNeuron * prevNeurons);

		deltaWeights[weightOffset + 
					 prevNeuron * prevNeurons + 
					 currNeuron] += neuronValues[prevOffset + prevNeuron] *
									   deltaValues[currOffset * currNeuron];

		batchDeltaWeights[weightOffset + 
					 prevNeuron * prevNeurons + 
					 currNeuron] += deltaWeights[weightOffset + 
					 							prevNeuron * prevNeurons + 
					 							currNeuron];
	}
};

void gpuBackpropagate(
					int layers,
					uint32_t * neurons,
					uint32_t * weightCounts,
					uint32_t * neuronOffsets,
					uint32_t * weightOffsets,
					float * expectedValues,
					float * neuronValues,
					float * weights,
					float * deltaValues,
					float * batchDeltaValues,
					float * deltaWeights,
					float * batchDeltaWeights,
					uint32_t totalWeightCount) {

	int blocksPerGrid;
	blocksPerGrid = 
	   		(neurons[layers - 1] + threadsPerBlock - 1) / threadsPerBlock;
	setDeltaExpect <<< blocksPerGrid, threadsPerBlock >>> (
		neuronOffsets[layers - 1],
		neurons[layers - 1],
		neuronValues,
		expectedValues,
		deltaValues);

	for(int i = layers-1; i > 0; i++) {
	   	blocksPerGrid = 
	   		(neurons[layers - 1] + threadsPerBlock - 1) / threadsPerBlock;

	   	getValueDeltas <<< blocksPerGrid, threadsPerBlock >>> (
	   		batchDeltaValues,
	   		deltaValues,
	   		weights,
	   		weightOffsets[i - 1],
	   		neuronOffsets[i],
	   		neuronOffsets[i - 1],
	   		neurons[i],
	   		neurons[i - 1]);

   		blocksPerGrid = (weightCounts[i] +
   						threadsPerBlock - 1) / threadsPerBlock;
	   	getWeightDeltas <<< blocksPerGrid, threadsPerBlock >>> (
			deltaValues,
			neuronValues,
			deltaWeights,
			batchDeltaWeights,
			neurons[i],
			neurons[i - 1],
			weightOffsets[i - 1],
			neuronOffsets[i - 1],
			neuronOffsets[i]);
	}

}

__global__ void updateParameters(
								float * deltaValues,
								float * deltaWeights,
								float * weights,
								float * biases,
								uint32_t weightCount,
								uint32_t neuronCount,
								float learningRate) {

   uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < weightCount) {
		biases[index] += deltaValues[index] * learningRate;
	}

	if(index < neuronCount) {
		weights[index] += deltaWeights[index] * learningRate;
	}
};

#define MAX(x, y) (x > y ? x : y)
void gpuUpdateParameters(float * deltaValues,
						float * deltaWeights,
						float * weights,
						float * biases,
						uint32_t totalWeightCount,
						uint32_t totalNeuronCount,
						float learningRate) {
	int blocksPerGrid = (MAX(totalWeightCount, totalNeuronCount) * 
   					threadsPerBlock - 1) / threadsPerBlock;

	updateParameters <<< blocksPerGrid, threadsPerBlock >>> (
		deltaValues,
		deltaWeights,
		weights,
		biases,
		totalWeightCount,
		totalNeuronCount,
		learningRate);
}
#undef MAX