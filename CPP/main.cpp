#include <iostream>
#include "MultiLayerPerceptron.h"
#include <math.h>
using namespace std;

int main() {

	const int NCE = 3; // numero de Neuronios na Camada de Entrada
	const int NCI = 4; // numero de Neuronios na Camada Intermediária
	
	double entradas[NCE][NCI] = {
		{0, 0, 1, 1},
		{0, 1, 0, 1},
		{1, 1, 1, 1} // Bias
	};
	
	/* 
	   Entradas corresponde as portas logicas XOR
	
	   A Matriz é transposta
	   
	   posição[0][0] = 0, XOR posição[1][0] = 0, Resposta: 0
	   posição[0][1] = 0, XOR posição[1][1] = 1, Resposta: 1
	   posição[0][2] = 1, XOR posição[1][2] = 0, Resposta: 1
	   posição[0][3] = 1, XOR posição[1][3] = 1, Resposta: 0
	
	   A ultima Linha da Matriz corresponde ao Bias
	*/
	
	double saidas_desejadas[NCI] = {0, 1, 1, 0};

/*	
    for(int i = 0; i < NCI; i++) {
		for(int j = 0; j < NCE; j++) {
			
			if(j < NCE - 1) {
				cout << entradas[j][i] << ", ";
			}else {
				cout << "Resposta => " << saidas_desejadas[i] << endl;
			}
			
		}
		cout << "\n";
	}
*/
	const double TAXA = 0.03;
	
	MultiLayerPerceptron net(NCE, NCI, TAXA);
	
	net.treinar(entradas, saidas_desejadas);
	
	//net.mostrar();
	
	double t1[3] = {0, 0, 1};
	double t2[3] = {1, 0, 1};
	double t3[3] = {0, 1, 1};
	double t4[3] = {1, 1, 1};
	
	net.classificar(t1);
	net.classificar(t2);
	net.classificar(t3);
	net.classificar(t4);
	
	return 0;
}
