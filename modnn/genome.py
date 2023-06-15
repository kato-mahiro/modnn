import random
import os
from modnn import Neuron

def read_config_file(file_path):
    config = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                try:
                    config[key] = int(value)
                except ValueError:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            config[key] = True
                        elif value.lower() == 'false':
                            config[key] = False
                        else:
                            config[key] = value
    return config

class genome:
    def __init__(self, config_path):
        self.config = read_config_file(config_path)
        self.input_num = self.config['INPUT_NUM']
        self.hidden_num = self.config['HIDDEN_NUM']
        self.modulatory_num = self.config['MODULATORY_NUM']
        self.output_num = self.config['OUTPUT_NUM']
        self.connection_num = self.config['CONNECTION_NUM']
        self.has_internal_state = self.config['HAS_INTERNAL_STATE']
        self.max_mod_depth = self.config['MAX_MOD_DEPTH']
        self.max_bias = self.config['MAX_BIAS']
        self.min_bias = self.config['MIN_BIAS']
        
        self.input_neurons = [Neuron(random.uniform(self.min_bias, self.max_bias)) for i in range(self.input_num)]
        self.hidden_neurons = [Neuron(random.uniform(self.min_bias, self.min_bias)) for i in range(self.hidden_num)]
        self.modulatory_neurons = [Neuron(random.uniform(self.min_bias, self.max_bias)) for i in range(self.modulatory_num)]
        self.output_neurons = [Neuron(random.uniform(self.min_bias, self.max_bias)) for id in range(self.output_num)]

if __name__ == '__main__':
    # 設定ファイルのパス
    pwd = os.path.dirname(os.path.abspath(__file__)) # このファイルのディレクトリの絶対パスを取得
    print(pwd)
    config_file_path = './tests/config.txt'

    # 設定ファイルを読み込む
    config = read_config_file(config_file_path)

    # 読み込んだ設定をプログラム内で利用する例
    hidden_num = config['HIDDEN_NUM']
    input_num = config['INPUT_NUM']
    output_num = config['OUTPUT_NUM']
    connection_num = config['CONNECTION_NUM']
    has_internal_state = config['HAS_INTERNAL_STATE']

    # 利用例として、読み込んだ設定を出力してみる
    print("Hidden neurons:", hidden_num)
    print("Input neurons:", input_num)
    print("Output neurons:", output_num)
    print("Number of connections:", connection_num)
    print("Has internal state:", has_internal_state)

    genome = genome(config_file_path)
    print(genome.input_neurons)
    print(genome.hidden_neurons)
    print(genome.output_neurons)