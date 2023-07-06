import random
import os
from modnn import Neuron
from modnn import Connection

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

class Genome:
    def __init__(self, config_path):
        self.config = read_config_file(config_path)
        self.input_num = self.config['INPUT_NUM']
        self.output_num = self.config['OUTPUT_NUM']
        self.normal_num = self.config['NORMAL_NUM']
        self.lv1_num = self.config['LV1_MODULATORY_NUM']
        self.lv2_num = self.config['LV2_MODULATORY_NUM']
        self.connection_num = self.config['CONNECTION_NUM']
        self.max_bias = self.config['MAX_BIAS']
        self.min_bias = self.config['MIN_BIAS']
        self.min_weight = self.config['MIN_WEIGHT']
        self.max_weight = self.config['MAX_WEIGHT']
        self.weight_upper_limit = self.config['WEIGHT_UPPER_LIMIT']
        self.weight_lower_limit = self.config['WEIGHT_LOWER_LIMIT']
        
        self.input_neurons = [Neuron(id =  i, bias = random.uniform(self.min_bias, self.max_bias)) for i in range(self.input_num)]
        self.output_neurons = [Neuron(id = i + self.input_num , bias = random.uniform(self.min_bias, self.max_bias)) for i in range(self.output_num)]
        self.normal_neurons = [Neuron(id = i + self.input_num + self.output_num, bias = random.uniform(self.min_bias, self.min_bias)) for i in range(self.normal_num)]
        self.lv1_neurons = [Neuron(id = i + self.input_num + self.output_num + self.normal_num, bias = random.uniform(self.min_bias, self.max_bias)) for i in range(self.lv1_num)]
        self.lv2_neurons = [Neuron(id = i + self.input_num + self.output_num + self.normal_num + self.lv1_num,  bias = random.uniform(self.min_bias, self.max_bias)) for i in range(self.lv2_num)]
        total_neuron_num = len(self.input_neurons) + len(self.output_neurons) + len(self.normal_neurons) + len(self.lv1_neurons) + len(self.lv2_neurons)
        self.connections = [ Connection(random.randint(0, total_neuron_num -1), random.randint(0, total_neuron_num -1), random.uniform(self.min_weight, self.max_weight)) for i in range(self.connection_num)]
    
    #ニューロンidからニューロンの種類を取得する
    def get_neuron_type(self, neuron_id):
        if neuron_id < self.input_num:
            return 'input'
        elif neuron_id < self.input_num + self.output_num:
            return 'output'
        elif neuron_id < self.input_num + self.output_num + self.neuron_num:
            return 'normal'
        elif neuron_id < self.input_num + self.output_num + self.neuron_num + self.lv1_num:
            return 'lv1'
        else:
            return 'lv2'

    #結合がルールに則っているかを判定する
    def is_valid_connection(self, connection):

        in_type = self.get_neuron_type(connection.from_neuron_id)
        out_type = self.get_neuron_type(connection.to_neuron_id)

        #出力ニューロンは結合元になれない
        if in_type == 'output':
            return False

        #入力ニューロンは結合先になれない
        elif out_type == 'input':
            return False

        #結合元と結合先が同じニューロンになれない
        elif connection.from_neuron_id == connection.to_neuron_id:
            return False

        #Lv.1の修飾ニューロンは通常ニューロン・出力ニューロン以外に結合できない
        elif in_type == 'lv1' and out_type != 'normal' and out_type != 'output':
            return False

        #Lv.2の修飾ニューロンはLv.1の修飾ニューロン以外に結合できない
        elif in_type == 'lv2' and out_type != 'lv1':
            return False
        
        else:
            return True

if __name__ == '__main__':
    # 設定ファイルのパス
    pwd = os.path.dirname(os.path.abspath(__file__)) # このファイルのディレクトリの絶対パスを取得
    print(pwd)
    config_file_path = './tests/config.txt'

    # 設定ファイルを読み込む
    config = read_config_file(config_file_path)

    # 読み込んだ設定をプログラム内で利用する例
    normal_num = config['NORMAL_NUM']
    input_num = config['INPUT_NUM']
    output_num = config['OUTPUT_NUM']
    connection_num = config['CONNECTION_NUM']

    # 利用例として、読み込んだ設定を出力してみる
    print("Normal neurons:", normal_num)
    print("Input neurons:", input_num)
    print("Output neurons:", output_num)
    print("Number of connections:", connection_num)

    genome = Genome(config_file_path)
    print(genome.input_neurons)
    print(genome.output_neurons)
    print(genome.normal_neurons)
    print(genome.lv1_neurons)
    print(genome.lv2_neurons)