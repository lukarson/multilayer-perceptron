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

	// metody tworz�ce macierze warto�ci Neurona
	Matrix* CreateValueMatrix();
	Matrix* CreateActValueMatrix();
	Matrix* CreateDerValueMatrix();

	// setter ustawiaj�cy warto�� wybranego Neurona w warstwie
	void SetNeuronValue(int i, double v) { neuronVector.at(i)->setValue(v); }

	//setter i getter wektora neuron�w, czyli warstwy
	void SetNeuronLayer(vector <Neuron*> neuronLayer) { neuronVector = neuronLayer; }
	vector<Neuron*> GetNeuronLayer() { return neuronVector; }

};

