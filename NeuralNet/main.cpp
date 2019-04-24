#include <iostream>
#include <vector>
#include <conio.h>
#include "NeuralNetwork.h"

using namespace std;

int main()
{ 
	vector<int> topology;
	char breaker = 'a';


	topology.push_back(4);
	topology.push_back(2);
	topology.push_back(3);
	topology.push_back(3);

	vector<double> input;

	input.push_back(1.0);
	input.push_back(0.5);
	input.push_back(1.0);
	input.push_back(2.5);
	
	vector<double> target;

	target.push_back(0.8);
	target.push_back(0.5);
	target.push_back(0.7);
	
	double bias = 1.0;
	double learningRate = 2;
	

	NeuralNetwork* neuNet = new NeuralNetwork(topology, bias, learningRate);

	neuNet->SetNetworkInput(input);
	neuNet->SetNetworkTargetOutput(target);

	while (breaker != 'q')
	{
		for (int i = 0; i < 50; i++)
		{
			cout << "Epoch: " << i + 1 << endl;
			neuNet->FeedForward();
			neuNet->ComputeCost();
			cout << "Total cost: " << neuNet->GetTotalCost() << endl << "=====================================================" << endl;
			neuNet->BackPropagation();
		}

		neuNet->PrintNetwork();	
		cout << endl << "Press 'enter' to start again or type 'q' to quit." << endl;
		fflush(stdin);
		breaker = getchar();
	}

	cout << "Export errors to txt file? (y/n): ";
	breaker = _getch();

	if (breaker == 'y')
		neuNet->ExportErrorsToFile();

	cout << endl;

	delete neuNet;

	system("pause");
	return 0;
}