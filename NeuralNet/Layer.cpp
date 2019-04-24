#include "Layer.h"



Layer::Layer(int size)
{
	sizeOfLayer = size;

	for (int i = 0; i < size; i++)
	{
		Neuron* n = new Neuron(0.0);
		neuronVector.push_back(n);
	}
}


Layer::~Layer()
{
}


Matrix * Layer::CreateValueMatrix()
{
	Matrix* m = new Matrix(1, neuronVector.size(), false);

	for (unsigned int i = 0; i < neuronVector.size(); i++)
		m->SetValue(0, i, neuronVector.at(i)->GetValue());

	return m;
}

Matrix * Layer::CreateActValueMatrix()
{
	Matrix* m = new Matrix(1, neuronVector.size(), false);

	for (unsigned int i = 0; i < neuronVector.size(); i++)
		m->SetValue(0, i, neuronVector.at(i)->GetActValue());

	return m;
}

Matrix * Layer::CreateDerValueMatrix()
{
	Matrix* m = new Matrix(1, neuronVector.size(), false);

	for (unsigned int i = 0; i < neuronVector.size(); i++)
		m->SetValue(0, i, neuronVector.at(i)->GetDerValue());

	return m;
}
