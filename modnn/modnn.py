import math
import pygraphviz as pgv

class Neuron:
    def __init__(self, id, bias):
        self.id = id
        self.output = 0.0
        self.m_output = 0.0
        self.bias = bias

class Connection:
    def __init__(self, from_id, to_id, weight):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight
        self.valid = True

class NN:
    def __init__(self, genome):
        self.genome = genome
        self.input_neurons = genome.input_neurons
        self.output_neurons = genome.output_neurons
        self.normal_neurons = genome.normal_neurons
        self.lv1_neurons = genome.lv1_neurons
        self.lv2_neurons = genome.lv2_neurons
        self.connections = genome.connections
        for c in self.connections:
            c.valid = self.is_valid_connection(c)

    def activate(self, inputs):
        # 入力ニューロンに値を設定
        for i in range(len(inputs)):
            self.input_neurons[i].output = inputs[i]

        # 隠れニューロンの値を計算
        for neuron in self.hidden_neurons:
            neuron_sum = 0.0
            # 入力からの結合を考慮
            for connection in self.connections:
                if connection.to_id == neuron.id:
                    neuron_sum += self.input_neurons[connection.from_id].output * connection.weight
            # シグモイド関数による活性化
            neuron.output = self.sigmoid(neuron_sum)

        # 出力ニューロンの値を計算
        outputs = []
        for neuron in self.output_neurons:
            neuron_sum = 0.0
            # 隠れからの結合を考慮
            for connection in self.connections:
                if connection.to_id == neuron.id:
                    neuron_sum += self.hidden_neurons[connection.from_id].output * connection.weight
            # シグモイド関数による活性化
            outputs.append(self.sigmoid(neuron_sum))

        return outputs

    def get_neuron_type(self, id):
        if id < self.genome.input_num:
            return "input"
        elif id < self.genome.input_num + self.genome.output_num:
            return "output"
        elif id < self.genome.input_num + self.genome.output_num + self.genome.normal_num:
            return "normal"
        elif id < self.genome.input_num + self.genome.output_num + self.genome.normal_num + self.genome.lv1_num:
            return "lv1"
        else:
            return "lv2"

    def is_valid_connection(self, connection):
        return True

    def visualize_graph(self):
        A = pgv.AGraph()
        for connection in self.connections:
            if connection.valid:
                A.add_edge(connection.from_id, connection.to_id)
        
        node_lsit = A.nodes()
        for node in node_lsit:
            id = int(node.get_name())
            print(id)
            node_type = self.get_neuron_type(id)
            if(node_type == "input"):
                node.attr['color'] = 'blue'
                node.attr['fillcolor'] = 'blue'
            elif(node_type == "output"):
                node.attr['color'] = 'red'
                node.attr['fillcolor'] = 'red'
            elif(node_type == "lv1"):
                node.attr['shape'] = 'box'
            elif(node_type == "lv2"):
                node.attr['shape'] = 'triangle'
            node.attr['style'] = 'filled'

            inode_subgraph = A.add_subgraph([i for i in range(self.genome.input_num)],rank='min')
            inode_subgraph = A.add_subgraph([i for i in range(self.genome.input_num, self.genome.input_num+self.genome.output_num)],rank='max')

        
        A.draw('graph.png', prog='dot')

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
