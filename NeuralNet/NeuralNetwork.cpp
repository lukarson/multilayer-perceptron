#include "NeuralNetwork.h"
#include <fstream>


NeuralNetwork::NeuralNetwork(vector<int> top, double bias, double learnRate)
{
	this->topology = top;
	this->bias = bias;
	this->learningRate = learnRate;
	this->topologySize = top.size();

	for (int i = 0; i < topologySize; i++)
	{
		Layer* l = new Layer(topology.at(i));
		layers.push_back(l);
	}

	for (int i = 0; i < (topologySize - 1); i++)
	{
		Matrix* m = new Matrix(topology.at(i), topology.at(i + 1), true); //generacja losowych wartoœci wag
		weightMatrices.push_back(m);
	}

	for (int i = 0; i < topology.at(topologySize - 1); i++)
		errorVector.push_back(0.00);

}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::SetNetworkInput(vector<double> in)
{
	input = in;

	for (unsigned int i = 0; i < input.size(); i++)
		layers.at(0)->SetNeuronValue(i, input.at(i));
}


void NeuralNetwork::FeedForward()
{
	Matrix* values = new Matrix(0, 0, false);
	Matrix* weights = new Matrix(0, 0, false);
	Matrix* newValues = new Matrix(0, 0, false);

	for (unsigned int i = 0; i < layers.size() - 1; i++)
	{
		values = (i != 0) ? GetActNeuronMatrix(i) : GetNeuronMatrix(i);

		weights = GetWeightMatrix(i);
		newValues = values;
		newValues = newValues->MultiplyMatrix(values, weights);
		
		for (int j = 0; j < newValues->GetNumOfColumns(); j++)
			SetNeuronValueAtLayer(i + 1, j, newValues->GetValue(0, j) + bias);
	}
	 
}

void NeuralNetwork::BackPropagation()
{
	int outputLayerIndex = layers.size() - 1;

	Matrix* derivedValuesYToZ = layers.at(outputLayerIndex)->CreateDerValueMatrix();
	Matrix* gradientsYToZ = new Matrix(1, layers.at(outputLayerIndex)->GetNeuronLayer().size(), false);
	Matrix* gradient;
	vector<Matrix*> newWeights;

	//-----------------------------------KROK 1--------------------------------------------
	//Policzenie propagacji dla warstwy LastHidden <----- Output
	
	
	//policzenie wektora gradientów warstw output i ostatniej hidden G = Z' * E ; E = (y - yHat)
	for (unsigned int i = 0; i < errorVector.size(); i++)
	{
		double der = derivedValuesYToZ->GetValue(0, i);
		double err = errorVector.at(i);
		double grad = der * err;
		gradientsYToZ->SetValue(0, i, grad); // G
	}

	
	//deltaW ostatniej warstwy hidden = Z^T * G
	int lastHiddenLayerIndex = outputLayerIndex - 1;
	
	Layer* lastHiddenLayer = layers.at(lastHiddenLayerIndex); 
	Matrix* Z = lastHiddenLayer->CreateActValueMatrix();
	
	Matrix* weightsOutputToHidden = weightMatrices.at(outputLayerIndex - 1); //  W
	Matrix* deltaOutputToHidden = new Matrix (0, 0, false); // dW
	deltaOutputToHidden = deltaOutputToHidden->MultiplyMatrix(Z->Transpose(), gradientsYToZ); // dW_out = Z^T * G

	

	//kopia macierzy dW, zaktualizowanie wag
	Matrix* newWeightsOutputToHidden = new Matrix(deltaOutputToHidden->GetNumOfRows(),
												  deltaOutputToHidden->GetNumOfColumns(),
												  false);
	
	//newW = W - dW
	for (int r = 0; r < deltaOutputToHidden->GetNumOfRows(); r++)
		for (int c = 0; c < deltaOutputToHidden->GetNumOfColumns(); c++)
		{
			double originalWeight = weightsOutputToHidden->GetValue(r, c);
			double deltaWeight = deltaOutputToHidden->GetValue(r, c);
			double difference = originalWeight - (learningRate * deltaWeight);
			newWeightsOutputToHidden->SetValue(r, c, difference);
		}

	
	newWeights.push_back(newWeightsOutputToHidden);
	gradient = gradientsYToZ;


	//------------------------------------------KROK 2----------------------------------------------
	//Policzenie propagacji dla warstw Input <--- Hidden(1) <--- Hidden(2) <--- ... <--- Hidden(n-1) <--- Hidden(n)

	for (int i = (outputLayerIndex - 1); i > 0; i--)
	{
		Layer* layer = layers.at(i);
		
		Matrix* derivedHidden = layer->CreateDerValueMatrix();
		Matrix* derivedGradients = new Matrix(1, layer->GetNeuronLayer().size(), false);
						
		Matrix* originalWeight = weightMatrices.at(i - 1);
		Matrix* weightMatrixT = weightMatrices.at(i)->Transpose();
		

		//G_prev * W_prev^T:
		derivedGradients = derivedGradients->MultiplyMatrix(gradient, weightMatrixT);

		//G_new = (G_prev * W_prev^T) x Z'		
		for (int r = 0; r < derivedGradients->GetNumOfRows(); r++)
			for (int c = 0; c < derivedGradients->GetNumOfColumns(); c++)
			{
				double product = derivedGradients->GetValue(r, c) * derivedHidden->GetValue(r, c);
				derivedGradients->SetValue(r, c, product);
			}
			
		
		Matrix* leftNeurons;
		
		if (i == 1)
		{
			leftNeurons = layers.at(0)->CreateValueMatrix();
		}
		else leftNeurons = layers.at(i - 1)->CreateActValueMatrix();
		
		
		// dW_in = X^T * G
		Matrix* deltaWeights = new Matrix(0, 0, false);
		deltaWeights = deltaWeights->MultiplyMatrix(leftNeurons->Transpose(), derivedGradients);

		//kopia deltaWeights przechowuj¹ca nowe wagi
		Matrix* newWeightsHidden = new Matrix(deltaWeights->GetNumOfRows(), deltaWeights->GetNumOfColumns(), false);

		
		//policzenie ró¿nicy (jak w kroku 1)
		for (int r = 0; r < deltaWeights->GetNumOfRows(); r++)
		{
			for (int c = 0; c < deltaWeights->GetNumOfColumns(); c++)
			{
				double weight = originalWeight->GetValue(r, c);
				double deltaWeight = deltaWeights->GetValue(r, c);
				double difference = weight - learningRate * deltaWeight;
				newWeightsHidden->SetValue(r, c, difference);
			}
		}
		
		//nowe gradienty dla kolejnej warstwy hidden
		gradient = derivedGradients;

		//nowa macierz wag hop do wektora nowych wag dla danej warstwy
		newWeights.push_back(newWeightsHidden);
	}

	reverse(newWeights.begin(), newWeights.end());
	weightMatrices = newWeights;

}

void NeuralNetwork::PrintNetwork()
{
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		cout << endl << "LAYER " << i << ":" << endl;
		if (i == 0)
		{
			Matrix* m = layers.at(i)->CreateValueMatrix();
			cout << endl;
			m->PrintMatrix();
		}
		else
		{
			Matrix* m = layers.at(i)->CreateActValueMatrix();
			cout << endl;
			m->PrintMatrix();
		}

		/* cout << "---------------------------------------" << endl;
		if (i < layers.size() - 1)
		{
			cout << "Weights for Layer " << i << endl;
			GetWeightMatrix(i)->PrintMatrix();
		}
		cout << "---------------------------------------" << endl; */
	}
}

void NeuralNetwork::ComputeCost()
{
	int outputLayerSize = layers.at(layers.size() - 1)->GetNeuronLayer().size();
	
	if (targetOutput.size() == 0)
	{
		cerr << "Network is missing target output" << endl;
		assert(false);
	}

	if (targetOutput.size() != outputLayerSize)
	{
		cerr << "Target output size different than output layer size! (" << targetOutput.size() << ", " << outputLayerSize << ")" << endl;
		assert(false);
	}

	error = 0.00;
	int outputLayerIndex = this->layers.size() - 1;
	vector<Neuron*> outputNeurons = this->layers.at(outputLayerIndex)->GetNeuronLayer();
	
	for (unsigned int i = 0; i < targetOutput.size(); i++)
	{
		double tempError = (outputNeurons.at(i)->GetActValue() - targetOutput.at(i));
		errorVector.at(i) = tempError;
		this->error += (tempError * tempError);
	}

	error = 0.5*error;

	allErrors.push_back(this->error);

}

void NeuralNetwork::ExportErrorsToFile()
{
	ofstream myfile;
	myfile.open("errors.txt");

	if (!myfile.good())
		cout << "Error opening file!\n\n";
	else
	{
		for (unsigned int i = 0; i < (allErrors.size() - 1); i++)
			myfile << allErrors.at(i) << endl;
		cout << endl;
	}
	
	myfile.close();
}
