# mpgraph

This repository contains an algorithm to compute the entire support solution space of a multi-penalty functional 

  1/2 ||A(u+v) - y||_2^2 + alpha ||u||_1 + beta/2 ||v||_2^2 -> min(u,v).
 
Here, the support solution space means that we are interested in all possible supports that are attained by the solution u_(alpha, beta) for some parameter combination (alpha, beta).

Concretely, this means the algorithm calculates the mapping 
  alpha, beta -> u_(alpha,beta), v_(alpha, beta) 
for all choices of (alpha, beta) up to a pre-defined sparsity level and in a pre-defined range [beta_min, beta_max]. In this it a related to the Lasso-path algorithm for the single-penalty Lasso or l1-functional that is able to efficiently calculate the mapping

  alpha -> u_(alpha) = argmin 1/2 ||Au - y||_2^2 + alpha ||u||_1.
  
In fact, the implementation of the algorithm relies on using this single-penalty variant with an efficient way of adding the second regularization parameter beta into the procedure. More details can be found in the paper '...' (soon).

More details on the method itself can be found in the paper, while the code contains many commentaries that explain the procedure.
