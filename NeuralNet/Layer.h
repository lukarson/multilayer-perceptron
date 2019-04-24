#pragma once

#include "Neuron.h"
#include "Matrix.h"

class Layer
{
private:
	int sizeOfLayer;
	vector<Neuron*> neuronVector; 

public:
	Layer(int size);
	~Layer();

	// metody tworz¹ce macierze wartoœci Neurona
	Matrix* CreateValueMatrix();
	Matrix* CreateActValueMatrix();
	Matrix* CreateDerValueMatrix();

	// setter ustawiaj¹cy wartoœæ wybranego Neurona w warstwie
	void SetNeuronValue(int i, double v) { neuronVector.at(i)->setValue(v); }

	//setter i getter wektora neuronów, czyli warstwy
	void SetNeuronLayer(vector <Neuron*> neuronLayer) { neuronVector = neuronLayer; }
	vector<Neuron*> GetNeuronLayer() { return neuronVector; }

};

