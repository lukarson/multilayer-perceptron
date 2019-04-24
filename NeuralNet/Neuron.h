#pragma once

#include <iostream>
#include <cstdlib>

using namespace std;

class Neuron
{
private:
	double value;
	double actValue;
	double derValue;

public:
	Neuron(double val);
	~Neuron();

	
	void Activation(); //liczy wartosc po aktywacji Neurona
	void Derivative(); //liczy pochodn¹ szybkiej funkcji sigmoidalnej
	
	void setValue(double v)
	{
		value = v;
		Activation();
		Derivative();
	}
	double GetValue() { return value; }
	double GetActValue() { return actValue; }
	double GetDerValue() { return derValue; }
};

