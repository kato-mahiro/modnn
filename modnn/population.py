import modnn
from modnn import utils
from modnn import Genome

class Population:
    def __init__(self, config_path):
        self.config = utils.read_config_file(config_path)
        self.genomes = []
        for i in range(self.config['POPULATION_SIZE']):
            self.genomes.append(Genome(self.config))