############ files ###########
- run.py
main function: 
select experiments: "testIris" and "testWine"; 
select actiation functions: "comb", "sigmoid", "tanh" and "leakyRelu" (default as mix)

- preprocessor.py
Preprocessor class: load data, encode categorical data, normalization, convert data to matrix for later training

- network.py
NNUtil and NeuralNetwork class: model training (backpropagation), predicting and calculating accuracy 

- /data/iris/*.txt
iris data files for experiment "testIris"

- /data/wine/*.txt
wine data files for experiment "testWine"

############# Usage ###########
python3 run.py -e <experiment> -a <activation> (need install numpy)

e.g.
"""
python3 run.py -e testIris -a sigmoid
"""
