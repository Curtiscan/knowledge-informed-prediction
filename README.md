# knowledge-informed-prediction

Authors: D. Lukashevich, N. Brilliantov, I.Tyukin

The project contains outline of knowledge-informed prediction, an approach to extrapolation of dynamic systems. The idea of the approach is narrowly described in an article Knowledge-Informed Neuro-Integrators for Aggregation Kinetics". In the heart of the concept we use a parametric family of functions, which is able to represent states of a dynamic system. If the set of its parameters is relatively small, this set can be used instead of full state of the dynamic system. Furthermore, if these parameters posses properties, that can be continued on target interval of prediction, we can utilize these properties to make accurate predictions of these parameters and consecutive accurate predictions of the states of the dynamic system.

In our work we consideres an aggregation process described with Smoluchowski system of equations. The project contains three scrips: 
1. ParameterRetriever.m establishes parametric family as a parametrizing neural network NN_p with two leanrable parameters.
2. ParameterPrediction.m provides prediction of modified versions of retrived parameters, that possess property of alternating derivatives.
3. SolutionReconstruction.m reconstructs prediction of solution of Smoluchowski system from predicted parameters and compares it to the real solution.
