#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#define END32SWAP(x) ((x>>24)&0xff) | \
					 ((x<<8)&0xff0000) | \
					 ((x>>8)&0xff00) | \
					 ((x<<24)&0xff000000)

int main() {
	int32_t magicNumber;

	FILE * labels = fopen("train-labels.idx1-ubyte", "rb");
	int32_t numLabels;

	fread(&magicNumber, 1, sizeof(int32_t), labels);
	fread(&numLabels, 1, sizeof(int32_t), labels);
	magicNumber = END32SWAP(magicNumber);
	numLabels = END32SWAP(numLabels);

	uint8_t label;
	float onHotEncoding[10];

	FILE * outLabels = fopen("labelData.bin", "wb");
	// Convert byte to one-hot encoding...
	for(int i = 0; i < numLabels; i++) {
		fread(&label, 1, sizeof(uint8_t), labels);
		for(int j = 0; j < 10; j++) {
			onHotEncoding[j] = (label == j ? 1 : 0);
		}
		fwrite(onHotEncoding, 10, sizeof(float), outLabels);
	}


	fclose(labels);
	fclose(outLabels);


	FILE * images = fopen("train-images.idx3-ubyte", "rb");


	int32_t numImages;
	int32_t columns;
	int32_t rows;

	fread(&magicNumber, 1, sizeof(int32_t), images);
	fread(&numImages, 1, sizeof(int32_t), images);
	fread(&columns, 1, sizeof(int32_t), images);
	fread(&rows, 1, sizeof(int32_t), images);
	magicNumber = END32SWAP(magicNumber);
	numImages = END32SWAP(numImages);
	columns = END32SWAP(columns);
	rows = END32SWAP(rows);

	uint8_t * imageData = (uint8_t *) malloc(columns * rows * sizeof(uint8_t));


	// Convert to float...
	float * imageDataF = (float *) malloc(columns * rows * sizeof(float));

	FILE * outImage = fopen("imageData.bin", "wb");
	for(int i = 0; i < numImages; i++) {
		fread(imageData, columns * rows, sizeof(uint8_t), images);
		for(int j = 0; j < columns * rows; j++) {
			imageData[j] = imageData[i] + 1;
			imageDataF[j] = (float) imageData[i] / (float) 255;
		}
		fwrite(imageDataF, columns * rows, sizeof(float), outImage);
	}



	fclose(images);
	fclose(outImage);

}
