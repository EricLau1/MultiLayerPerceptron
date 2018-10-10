#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

class MultiLayerPerceptron {
	
	private:
		
		double TAXA_APRENDIZADO; // Melhores valores para taxa de aprendizado fica entre 0.1 e 0.7 
		
		double** w1c; // pesos da 1ªcamada
		double* w2c; // pesos da 2ªcamada
		int nci;	  // numero de Neuronios da Camada Intermediaria
		int nce;	  // numero de Neuronios na Camada de Entrada
		
		int epocas;
		
	public:
		
		MultiLayerPerceptron(int nce, int nci, const double TAXA) {
			
			this->TAXA_APRENDIZADO = TAXA;
			
			this->epocas = 0;
			
			this->nce = nce;
			
			this->nci = nci;
			
			iniciarPesos();
						
		}
		
		void mostrar();
		
		void iniciarPesos();
		
		void treinar(double[][4], double*);
		
		double sigmoid(double);
		
		double* run1(double[][4], int);
		double run2(double*);
		
		double gradiente(double, double);
	
		void retropropagacao(double[][4], double*, double, int);
		
		void classificar(double*);
};

void MultiLayerPerceptron::mostrar() {
	
	cout << "MultiLayer Perceptron\n" << endl;
	
	cout << "TAXA DE APRENDIZADO: " << this->TAXA_APRENDIZADO << endl;
	cout << "Epocas: " << this->epocas << endl;
	cout << "Numero de Neuronios na Camada de Entrada: " << this->nce << endl;
	cout << "Numero de Neuronios na camada Intermediaria: " << this->nci << endl;
	
	cout << "\nInicializando pesos da primeira camada...\n" << endl;
	for(int i = 0; i < this->nci; i++) {
		for(int j = 0; j < this->nce; j++) {
			cout << "w1: " << this->w1c[i][j] << "\t";
		}
		cout << endl;
	}
	cout << "\n";
	
	cout << "Inicializando pesos da segunda camada...\n" << endl;
	for(int i = 0; i < this->nci; i++) {
		cout << "w2: " << this->w2c[i] << endl; 
	}
	cout << "\n";	
}

void MultiLayerPerceptron::iniciarPesos() {
	
	srand(time(NULL));
	
	
	// Inicializando Pesos da primeira camada
	this->w1c = (double**)malloc(this->nci * sizeof(*this->w1c)); // Alocando Memoria para as linhas

	for(int i = 0; i < this->nci; i++) {
		this->w1c[i] = (double*)malloc(this->nce * sizeof(this->w1c[i])); // Alocando Memoria para as colunas
		for(int j = 0; j < this->nce; j++) {
			this->w1c[i][j] = (double) rand() / RAND_MAX;
		}
	}
	
	// Inicializando Pesos da Segunda Camada
	this->w2c = (double*)malloc(this->nci * sizeof(double));
	for(int i = 0; i < this->nci; i++) {
		this->w2c[i] = (double) rand() / RAND_MAX;
	}
	
}

double MultiLayerPerceptron::sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double* MultiLayerPerceptron::run1(double entradas[][4], int i) {
	
	// vetor de saidas da primeira camada
	double* saidas = (double*) malloc (this->nci * sizeof(double));
		
	//cout << "Saidas da primeira camada..." << endl;
	for(int j = 0; j < this->nci; j++) {
		
		double soma = 0.0;
		
		for(int k = 0; k < this->nce; k++) {
			soma += entradas[k][i] * this->w1c[j][k];			
		}	
			
		saidas[j] = sigmoid(soma);
		
		//cout << "output: " << saidas[j] << endl;	
	}
	
	return saidas;
}

double MultiLayerPerceptron::run2(double* e2c) {
	
	double soma = 0.0;
	// Somatoria da multiplicação das entradas da segunda camada pelos pesos da segunda camada
	for(int i = 0; i < this->nci; i++) {
		soma += e2c[i] * this->w2c[i]; 
	}
	return sigmoid(soma);
}

double MultiLayerPerceptron::gradiente(double saida_atual, double erro) {
	return saida_atual * (1 - saida_atual) * erro;
}

void MultiLayerPerceptron::retropropagacao(double entradas[][4], double* e2c, double gradiente, int i) {
	
	// Retropropagação de Erro pela Segunda Camada 
	for(int j = 0; j < this->nci; j++) {
		// Correção dos pesos da segunda camada
		this->w2c[j] +=  this->TAXA_APRENDIZADO * e2c[j] * gradiente;
	}
	
	// Retropropagação de Erro pela Primeira Camada
	for(int j = 0; j < this->nci - 1; j++) {
		
		// derivada da função de transferencia
		double derivada = e2c[j] * (1 - e2c[j]);
		
		double sigma = derivada * (this->w2c[j] * gradiente);
		
		for(int k = 0; k < this->nce; k++) {
			// Correção dos pesos da Primeira Camada
			this->w1c[j][k] += this->TAXA_APRENDIZADO * sigma * entradas[k][i];
		}
	}
}

void MultiLayerPerceptron::treinar(double entradas[][4], double* saidas_desejadas) {
	
	double erro = 1.0;
	
	double* e2c = (double*)malloc(this->nci * sizeof(double)); // valores de entradas para a camada intermediaria de neuronios
	
	while((erro > 0.05) && (this->epocas < 100000)) {
		
		for(int i = 0; i < this->nci; i++) {
			
			e2c = run1(entradas, i);
			
			/*
			
			// Visualizar valores de saida da primeira camada e entradas da segunda camada
			
			cout << "\nEntradas da segunda camada...\n" << endl;
			for(int j = 0; j < 4; j++) {
				cout << "input: " << e2c[j] << ", "; 
			}
			cout << "\n"; 
			
			*/
			
			// Valor de Saida da segunda camada
			double saida_atual = run2(e2c);
			//cout << "saida atual: " << saida_atual << endl;
			
			// Calculando o Erro
			erro = saidas_desejadas[i] - saida_atual;
			
			// Calculando o gradiante de retropropagação
			double g = gradiente(saida_atual, erro);
			//cout << "gradiente: " << g << endl;
			
			retropropagacao(entradas, e2c, g, i);
			/*
			if(this->epocas % 100 == 0) {
				printf("Erro: %.10f\n", erro);
			} */
						
		}
		erro = (erro < 0)? erro * -1: erro;
		this->epocas++;
	}
}

void MultiLayerPerceptron::classificar(double* entradas) {
	
	if(this->epocas > 999999) {
		cout << "A Rede Neural nao convergiu... E necessario modificar a estrutura da Rede." << endl;
	}
	else {
		
		double saidas[this->nci]; // saidas da primeira camada
		
		for(int j = 0; j < this->nci; j++) {
			double soma = 0.0;
			for(int k = 0; k < this->nce; k++) {
				soma += entradas[k] * this->w1c[j][k];
			}
			saidas[j] = sigmoid(soma);
		}
		
		double y = run2(saidas);
		
		string resposta = (y < 0.5)? "False" : "True";
		
		cout << resposta << endl;
	}
}


