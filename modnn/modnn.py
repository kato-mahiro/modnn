import math
import pygraphviz as pgv

class Neuron:
    def __init__(self, id, bias):
        self.id = id
        self.output = 0.0
        self.m_output = 0.0
        self.modulated = 0.0
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
        assert(len(inputs) == self.genome.input_num)

        # 入力ニューロンに値を設定
        for i in range(len(inputs)):
            self.input_neurons[i].output = inputs[i]
        
        # 通常ニューロンの値を計算
        for n in self.normal_neurons:
            neuron_sum = n.bias
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "input"):
                            neuron_sum += self.input_neurons[connection.from_id].output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "normal"):
                            neuron_sum += self.normal_neurons[connection.from_id - self.genome.input_num -self.genome.output_num].output * connection.weight
            #シグモイド関数による活性化
            n.output = self.sigmoid(neuron_sum)
            
        # lv.1 ニューロンの値を計算
        for n in self.lv1_neurons:
            neuron_sum = n.bias
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "input"):
                            neuron_sum += self.input_neurons[connection.from_id].output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "normal"):
                            neuron_sum += self.normal_neurons[connection.from_id - self.genome.input_num -self.genome.output_num].output * connection.weight
            #シグモイド関数による活性化
            n.m_output = self.sigmoid(neuron_sum)

        # lv.2 ニューロンの値を計算
        for n in self.lv2_neurons:
            neuron_sum = n.bias
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "input"):
                            neuron_sum += self.input_neurons[connection.from_id].output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "normal"):
                            neuron_sum += self.normal_neurons[connection.from_id - self.genome.input_num -self.genome.output_num].output * connection.weight
            #シグモイド関数による活性化
            n.m_output = self.sigmoid(neuron_sum)

        # 出力ニューロンの値を計算
        outputs = []
        for n in self.output_neurons:
            neuron_sum = n.bias
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "input"):
                            neuron_sum += self.input_neurons[connection.from_id].output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "normal"):
                            neuron_sum += self.normal_neurons[connection.from_id - self.genome.input_num - self.genome.output_num].output * connection.weight
            # シグモイド関数による活性化
            n.output = self.sigmoid(neuron_sum)
            outputs.append(n.output)
        return outputs

    def weight_update(self):
        #activate()の実行後に実行する

        #各ニューロンがどの程度修飾されているか(modulated)を計算
        # 通常ニューロンのmodulated値を計算
        for n in self.normal_neurons:
            modulated_sum = 0.0
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "lv1"):
                            modulated_sum += self.lv1_neurons[connection.from_id - self.genome.input_num - self.genome.output_num - self.genome.normal_num].m_output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "lv2"):
                            modulated_sum += self.lv2_neurons[connection.from_id - self.genome.input_num -self.genome.output_num - self.genome.normal_num - self.genome.lv1_num].m_output * connection.weight
            n.modulated = modulated_sum

        # lv.1 ニューロンのmodulated値を計算
        for n in self.lv1_neurons:
            modulated_sum = 0.0
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "lv1"):
                            print("ERROR: lv1ニューロンがlv1ニューロンに修飾されている")
                            exit()
                            modulated_sum += self.lv1_neurons[connection.from_id - self.genome.input_num - self.genome.output_num - self.genome.normal_num].m_output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "lv2"):
                            modulated_sum += self.lv2_neurons[connection.from_id - self.genome.input_num -self.genome.output_num - self.genome.normal_num - self.genome.lv1_num].m_output * connection.weight
            n.modulated = modulated_sum

        # lv.2 ニューロンのmodulated値を計算
        for n in self.lv2_neurons:
            n.modulated = 0.0 #lv2ニューロンは修飾されない

        #出力ニューロンのmodulated値を計算
        for n in self.output_neurons:
            modulated_sum = 0.0
            for connection in self.connections:
                if(connection.valid):
                    if connection.to_id == n.id:
                        if (self.get_neuron_type(connection.from_id) == "lv1"):
                            modulated_sum += self.lv1_neurons[connection.from_id - self.genome.input_num - self.genome.output_num - self.genome.normal_num].m_output * connection.weight
                        elif(self.get_neuron_type(connection.from_id) == "lv2"):
                            modulated_sum += self.lv2_neurons[connection.from_id - self.genome.input_num -self.genome.output_num - self.genome.normal_num - self.genome.lv1_num].m_output * connection.weight
            n.modulated = modulated_sum

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
        #再帰的(自分に戻ってくる)結合は無効化
        if connection.from_id == connection.to_id:
            return False
        #入力ノードに入ってくる結合は無効化
        elif self.get_neuron_type(connection.to_id) == "input":
            return False
        #出力ノードから出ていく結合は無効化
        elif self.get_neuron_type(connection.from_id) == "output":
            return False

        #通常ニューロンどうしの結合は、idの小さいほうから大きいほうへのみ有効
        elif self.get_neuron_type(connection.from_id) == "normal" and self.get_neuron_type(connection.to_id) == "normal":
            if connection.from_id > connection.to_id:
                return False
            else:
                return True
        #lv1修飾ニューロンは、通常ニューロン・出力ニューロン以外に結合できない
        elif self.get_neuron_type(connection.from_id) == "lv1"\
             and self.get_neuron_type(connection.to_id) != "normal" \
             and self.get_neuron_type(connection.to_id) != "output":
            return False

        #lv2修飾ニューロンは、lv1修飾ニューロン以外に結合できない
        elif self.get_neuron_type(connection.from_id) == "lv2" and self.get_neuron_type(connection.to_id) != "lv1":
            return False

        else:
            return True


    def visualize_graph(self):
        A = pgv.AGraph(directed=True)
        for connection in self.connections:
            if connection.valid:
                if(self.get_neuron_type(connection.from_id) == "lv2" or self.get_neuron_type(connection.to_id) == "lv2"):
                    color = "purple"
                elif(self.get_neuron_type(connection.from_id) == "lv1" or self.get_neuron_type(connection.to_id) == "lv1"):
                    color = "blue"
                else:
                    color = "black"
                A.add_edge(connection.from_id, connection.to_id, color=color)
        
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
