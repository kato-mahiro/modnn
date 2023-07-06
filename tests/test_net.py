import modnn

genome = modnn.Genome('./config.txt')
print(genome)
net = modnn.NN(genome)
net.visualize_graph()