/* 
 * Autor: Eric Lau de Oliveira
 * 
 * 17-06-2018 21H18m
 * 
 * 	MultLayerPerceptron
 * 
 * */
public class Main {

	public static void main(String[] args) {
		
		final int NCE = 3; // numero de neuronios da camada de entrada
		final int NCI = 4; // numero de neuronios da camada intermediaria
		
		/* 
		 * Matriz Transposta representando as entradas do XOR
		 *
		 * [0][0] = 0, [1][0] = 0 resposta => 0
		 * [0][1] = 0, [1][1] = 1 resposta => 1
		 * [0][2] = 1, [1][2] = 0 resposta => 1
		 * [0][3] = 1, [1][3] = 1 resposta => 0
		 * 
		 * A terceira linha representa o valor de bias
 		 */
		
		double[][] entradas = {
			new double[]{0, 0, 1, 1},
			new double[]{0, 1, 0, 1},
			new double[]{1, 1, 1, 1} // BIAS
		};
	
		double saidas[] = {0, 1, 1, 0};
		
		final double TAXA = 0.03d;
		
		MLP net = new MLP(NCE, NCI, TAXA);
		
		net.treinar(entradas, saidas);
		
		double t1[] = {0, 0, 1};
		double t2[] = {1, 0, 1};
		double t3[] = {0, 1, 1};
		double t4[] = {1, 1, 1};
		
		net.classificar(t1);
		net.classificar(t2);
		net.classificar(t3);
		net.classificar(t4);
		
	}

}
