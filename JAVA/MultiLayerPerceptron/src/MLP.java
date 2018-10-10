
public class MLP {
	
	private double TAXA_APRENDIZADO; // Melhores valores para taxa de aprendizado fica entre 0.1 e 0.7 
	
	private double[][] p1c; // pesos da primeira camada 
	private double[]   p2c; // pesos da segunda camada
	
	private int nci; //numero de neuronios da camada intermediária
	private int nce; //numero de neuronios da camada de entrada
	
	int epocas;
	
	public MLP(int nce, int nci, double TAXA) {
		
		this.nce = nce;
		this.nci = nci;
		
		this.TAXA_APRENDIZADO = TAXA;
		
		this.epocas = 0;
		
		iniciarPesos();
	}
	
	private void iniciarPesos() {
		
		System.out.println("Inicializando pesos da primeira camada...");
		
		this.p1c = new double[this.nci][this.nce];
		
		//System.out.println("Linhas: " + this.p1c.length);
		//System.out.println("Colunas: " + this.p1c[0].length);
		
		
		for(int i = 0; i < this.p1c.length; i++) {
			
			for(int j = 0; j < this.p1c[0].length; j++) {
				
				this.p1c[i][j] = Math.random();
				
				//System.out.printf("w: %.8f\t", this.p1c[i][j] );
				
			} // fim for j
			
			//System.out.println("\n");
			
		} // fim for i
		
		System.out.println("Inicializando pesos da segunda camada...");
		
		this.p2c = new double[this.nci];
		
		for(int i = 0; i < this.p2c.length; i++) {
			this.p2c[i] = Math.random();
		}
		
		System.out.println("Pesos inicializados com sucesso!");
		
	} // fim metodo
	
	public double[] run1(double[][] entradas, int i) {
		
		//vetor de saidas da primeira camada
		double saidas[] = new double[this.nci];
		
		for(int j = 0; j < this.nci; j++) {
			
			double soma = 0.0d;
			
			for(int k = 0; k < this.nce; k++) {
				
				soma += entradas[k][i] * this.p1c[j][k];
				
			} // fim for k
			
			saidas[j] = ActivationFunction.Sigmoid(soma);
			
		} // fim for j
		
		return saidas;
		
	}// fim método
	
	public double run2(double[] e2c) {
		
		double soma = 0.0d;
		// somatorio das saidas da primeira camada com os pesos da segunda camada
		for(int i = 0; i < this.nci; i++) {
			soma += e2c[i] * this.p2c[i];
		}
		return ActivationFunction.Sigmoid(soma);
		
	} // fim matodo

	public double gradiente(double saida_atual, double erro) {
		
		return saida_atual * ( 1 - saida_atual ) * erro;
		
	} // fim metodo
	
	private void retropropagacao(double[][] entradas, double[] e2c, double gradiente, int i ) {
		
		// Retropropagação de erro pela segunda camada
		for(int j = 0; j < this.nci; j++) {
			// correção dos pesos da segunda camada
			this.p2c[j] += this.TAXA_APRENDIZADO * e2c[j] * gradiente;
		}
		
		// Retropropagação de erro pela primeira camada
		for(int j = 0; j < this.nci - 1; j++) {
			
			// derivada da função de transferencia
			double derivada = e2c[j] * (1 - e2c[j]);
			
			double sigma = derivada * (this.p2c[j] * gradiente);
			
			for(int k = 0; k < this.nce; k++) {
				
				// correção dos pesos da primeira camada
				this.p1c[j][k] += this.TAXA_APRENDIZADO * sigma * entradas[k][i];
			
			} // fim for k 
			
		} // fim for j
		
	} // fim metodo
	
	public void treinar(double[][] entradas, double[] saidas_desejadas) {
		
		double erro = 1.0d;
		
		double[] e2c = new double[this.nci];
		
		while( (erro > 0.05) && (this.epocas < 100000) ) {
			
			for( int i = 0; i < this.nci; i++) {
				
				// inicializando as entradas da camada intermediaria de neuronios
				e2c = this.run1(entradas, i);
				
				// valor de saida da segunda camada de neuronios
				double saida_atual = this.run2(e2c);
				
				// calculando erro
				erro = saidas_desejadas[i] - saida_atual;
				
				// Calculando o gradiente de retropropagação
				double g = this.gradiente(saida_atual, erro);
				
				this.retropropagacao(entradas, e2c, g, i);
				
				if(this.epocas % 100 == 0) {
					System.out.printf("Erro: %.10f\n", erro);
				}
				
			} // fim for i
			
			erro = Math.abs(erro); // valor absoluto do erro
			this.epocas++;
			
		} // fim while
		
	} // fim metodo
	
	public void classificar(double[] entradas) {
		
		if(this.epocas > 999999) {
			
			System.out.println("Rede neural não convergiu...\n" +
			"Modifique a estrutura da rede.");
			
		} else {
			
			System.out.println("Classificando...");
			
			double[] saidas = new double[this.nci];
			
			for(int j = 0; j < this.nci; j++) {
				
				
				double soma = 0.0d;
				
				for(int k = 0; k < this.nce; k++) {
					
					soma += entradas[k] * this.p1c[j][k];
					
				} // fim for l
				
				
				saidas[j] = ActivationFunction.Sigmoid(soma);
				
			} // fim for j
			
			
			double y = this.run2(saidas);
			
			int r = (y < 0.5)? 0 : 1;
			
			System.out.println(r);
			
		} // fim else
		
	} // fim metodo
	
	
} // fim classe
