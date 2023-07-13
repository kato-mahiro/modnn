import random
import modnn
from modnn import utils
from modnn import Genome

class Population:
    def __init__(self, config_path):
        self.config = utils.read_config_file(config_path)
        self.genomes = []
        for i in range(self.config['POPULATION_SIZE']):
            self.genomes.append(Genome(self.config))
        
    def run(self, eval_fitness):
        for g in range(self.config['GENERATION_NUM']):
            for genome in self.genomes:
                net = modnn.NN(genome)
                genome.fitness = eval_fitness(net)
            self.genomes = self.evo(self.genomes)
            print(g)

    def evo(self, genomes):
        sorted_genomes = sorted(genomes, key=lambda x: x.fitness, reverse=True)
        selection_num = int(self.config['SELECTION_RATE'] * self.config['POPULATION_SIZE'])
        selected_genomes = sorted_genomes[:selection_num]
        next_genomes = []
        for i in range(self.config['ELITE_SIZE']):
            next_genomes.append(selected_genomes[i])
        for i in range(self.config['POPULATION_SIZE'] - self.config['ELITE_SIZE']):
            parent1 = random.choice(selected_genomes)
            parent2 = random.choice(selected_genomes)
            child = self.cross_over(parent1, parent2)
            child = self.mutate(child)
            next_genomes.append(child)
        return sorted_genomes

    def cross_over(self, parent1, parent2):
        child = Genome(self.config)

        for i in range(len(child.connections)):
            if random.random() < 0.5:
                child.connections[i] = parent1.connections[i]
            else:
                child.connections[i] = parent2.connections[i]

        for i in range(len(child.input_neurons)):
            if random.random() < 0.5:
                child.input_neurons[i] = parent1.input_neurons[i]
            else:
                child.input_neurons[i] = parent2.input_neurons[i]

        for i in range(len(child.output_neurons)):
            if random.random() < 0.5:
                child.output_neurons[i] = parent1.output_neurons[i]
            else:
                child.output_neurons[i] = parent2.output_neurons[i]
        
        for i in range(len(child.normal_neurons)):
            if random.random() < 0.5:
                child.normal_neurons[i] = parent1.normal_neurons[i]
            else:
                child.normal_neurons[i] = parent2.normal_neurons[i]
        
        for i in range(len(child.lv1_neurons)):
            if random.random() < 0.5:
                child.lv1_neurons[i] = parent1.lv1_neurons[i]
            else:
                child.lv1_neurons[i] = parent2.lv1_neurons[i]
            
        for i in range(len(child.lv2_neurons)):
            if random.random() < 0.5:
                child.lv2_neurons[i] = parent1.lv2_neurons[i]
            else:
                child.lv2_neurons[i] = parent2.lv2_neurons[i]
        
        if random.random() < 0.5:
            child.a = parent1.a
        else:
            child.a = parent2.a
        if random.random() < 0.5:
            child.b = parent1.b
        else:
            child.b = parent2.b
        if random.random() < 0.5:
            child.c = parent1.c
        else:
            child.c = parent2.c
        if random.random() < 0.5:
            child.d = parent1.d
        else:
            child.d = parent2.d
        
        return child
    
    def mutate(self, genome):
        mutate_rate = self.config['MUTATION_RATE']
        for i in range(len(genome.connections)):
            if random.random() < mutate_rate:
                genome.connections[i].weight += random.uniform(-1, 1)
        
        for i in range(len(genome.input_neurons)):
            if random.random() < mutate_rate:
                genome.input_neurons[i].bias += random.uniform(-1, 1)
        
        for i in range(len(genome.output_neurons)):
            if random.random() < mutate_rate:
                genome.output_neurons[i].bias += random.uniform(-1, 1)
        
        for i in range(len(genome.normal_neurons)):
            if random.random() < mutate_rate:
                genome.normal_neurons[i].bias += random.uniform(-1, 1)
        
        for i in range(len(genome.lv1_neurons)):
            if random.random() < mutate_rate:
                genome.lv1_neurons[i].bias += random.uniform(-1, 1)
        
        for i in range(len(genome.lv2_neurons)):
            if random.random() < mutate_rate:
                genome.lv2_neurons[i].bias += random.uniform(-1, 1)
        
        if random.random() < mutate_rate:
            genome.a += random.uniform(-1, 1)
        
        if random.random() < mutate_rate:
            genome.b += random.uniform(-1, 1)
        
        if random.random() < mutate_rate:
            genome.c += random.uniform(-1, 1)

        if random.random() < mutate_rate:
            genome.d += random.uniform(-1, 1)
                
        return genome