#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <assert.h>


using namespace std;

class Matrix
{
private:
	int numOfColumns;
	int numOfRows;
	
	vector< vector<double> > matrix;

public:
	Matrix(int rows, int columns, bool isRandom);
	~Matrix();

	Matrix* Transpose();
	Matrix* MultiplyMatrix(Matrix* a, Matrix* b);

	// generowanie losowej liczby od 0 do 1
	double RNG();

	// setter i getter elementu macierzy
	void SetValue(int r, int c, double v) { matrix.at(r).at(c) = v; }
	double GetValue(int r, int c) { return matrix.at(r).at(c); }
	
	// funkcja pokazuj¹ca macierz w konsoli
	void PrintMatrix();

	//gettery wymiarów
	int GetNumOfRows() { return numOfRows; }
	int GetNumOfColumns() { return numOfColumns; }
};

