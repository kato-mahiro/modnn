import random
from modnn import population

def eval_fitness(net):
    return random.random()

p = population.Population('./config.txt')
print(p.genomes)

p.run(eval_fitness)