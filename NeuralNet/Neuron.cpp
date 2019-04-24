#include "Neuron.h"


Neuron::Neuron(double v)
{
	value = v;
	Activation();
	Derivative();
}


Neuron::~Neuron()
{
}

// f(x) = x / (1 + |x|)
void Neuron::Activation()
{
	actValue = value / (1 + abs(value));
}

// f'(x) = f(x)(1 - f(x))
void Neuron::Derivative()
{
	derValue = actValue * (1 - actValue);
}
