#include "Matrix.h"



Matrix::Matrix(int rows, int columns, bool isRandom)
{
	numOfRows = rows;
	numOfColumns = columns;
	double v;
	
	for (int i = 0; i < numOfRows; i++)
	{
		vector<double> values;

		for (int j = 0; j < numOfColumns; j++)
		{
			if (isRandom)
				v = RNG();
			else 
				v = 0.00;

			values.push_back(v);
		}

		matrix.push_back(values);
	}
}

Matrix::~Matrix()
{
}

Matrix * Matrix::Transpose()
{
	Matrix* m = new Matrix(numOfColumns, numOfRows, false);

	for (int i = 0; i < numOfRows; i++)
		for (int j = 0; j < numOfColumns; j++)
			m->SetValue(j, i, GetValue(i, j));

	return m;
}

Matrix * Matrix::MultiplyMatrix(Matrix * a, Matrix * b)
{
	double product, cValue;
	
	if (a->GetNumOfColumns() != b->GetNumOfRows())
	{
		cerr << "A columns = " << a->GetNumOfColumns() << " and B rows = " << b->GetNumOfRows() << "! Error!" << endl;
		assert(false);
	}

	Matrix* c = new Matrix(a->GetNumOfRows(), b->GetNumOfColumns(), false);

	for (int i = 0; i < a->GetNumOfRows(); i++)
	{
		for (int j = 0; j < b->GetNumOfColumns(); j++)
		{
			for (int k = 0; k < b->GetNumOfRows(); k++)
			{
				product = a->GetValue(i, k) * b->GetValue(k, j);
				cValue = c->GetValue(i, j) + product;
				c->SetValue(i, j, cValue);
			}
		}
	}

	return c;
}

double Matrix::RNG()
{
	random_device dev;
	mt19937 gen(dev());
	uniform_real_distribution<> dis(0, 1);
	return dis(gen);
}

void Matrix::PrintMatrix()
{
	for (int i = 0; i < numOfRows; i++)
	{
		for (int j = 0; j < numOfColumns; j++)
			cout << fixed << matrix.at(i).at(j) << "\t";
		cout << endl;
	}
}
