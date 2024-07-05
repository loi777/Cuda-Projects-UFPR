//  Utilização do programa:
//  usage: ./cudaReduceMax <nTotalElements> nR
//  onde:
//          <nTotalElements> e' o número de floats do vetor de entrada
//          nR               número de Repeticoes, ou seja,
//                             quantas vezes é chamado cada kernel
//                             e medindo a média de tempo gasto por cada kernel.
//                             assim é possivel aumentar a precisao de medidas
//                             aplicando sincronizaçao antes e depois da repeticao
//                             de cada tipo de kernel. 
//                             (fazer medidas independentes por tipo de kernel)
//                             a repeticao pode ser feita com o mesmo vetor de entrada
//                             SEM gerar novo a cada iteracao 