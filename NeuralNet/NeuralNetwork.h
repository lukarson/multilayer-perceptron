#pragma once

#include "Matrix.h"
#include "Layer.h"

using namespace std;

class NeuralNetwork
{
private:

	vector<int> topology; //wektor rozmiar�w kolejnych warstw (w Neuronach)
	int topologySize;

	vector<Layer*> layers; //wektor warstw Neuron�w

	vector<Matrix*> weightMatrices; //wektor macierzy wag (synaps), sizeof(weightMatrices) = sizeof(topology) - 1

	vector<double> input; //wektor warto�ci wej�ciowych

	vector<double> targetOutput;
	
	double error;

	vector<double> errorVector;

	vector<double> allErrors;

	double bias = 1.0;
	double learningRate = 0.5;

public:
	NeuralNetwork(vector<int> top, double bias, double learnRate);
	~NeuralNetwork();

	//ustawia warto�ci Neuron�w pierwszej warstwy wektorem 'in'
	void SetNetworkInput(vector<double> in);

	//ustawia oczekiwane warto�ci wyj�cia (referencja dla b��d�w)
	void SetNetworkTargetOutput(vector<double> target) { this->targetOutput = target; }

	//metoda feedForward
	void FeedForward();

	//propagacja wsteczna
	void BackPropagation();

	//pokazuje topologi� sieci w konsoli
	void PrintNetwork();


	//funkcje tworz�ce macierze
	Matrix* GetNeuronMatrix(int index) { return layers.at(index)->CreateValueMatrix(); }
	Matrix* GetActNeuronMatrix(int index) { return layers.at(index)->CreateActValueMatrix(); };
	Matrix* GetDerNeuronMatrix(int index) { return layers.at(index)->CreateDerValueMatrix(); };
	Matrix* GetWeightMatrix(int index) { return weightMatrices.at(index); };

	void SetNeuronValueAtLayer(int layerIndex, int neuronIndex, double value) { layers.at(layerIndex)->SetNeuronValue(neuronIndex, value); }

	//pobiera warto�ci Neuron�w wyj�ciowych, liczy b��d, wstawia go do wektora b��d�w oraz oblicza koszt
	void ComputeCost();
	double GetTotalCost() { return error; }

	void ExportErrorsToFile();
};

