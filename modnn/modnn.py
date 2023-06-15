import math

class Neuron:
    def __init__(self, bias):
        self.output = 0.0
        self.m_output = 0.0
        self.bias = bias

class Connection:
    def __init__(self, from_id, to_id, weight):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight

class NeuralNetwork:
    def __init__(self, input_neurons, output_neurons, hidden_neurons, connections):
        self.input_neurons = [Neuron(id) for id in input_neurons]
        self.output_neurons = [Neuron(id) for id in output_neurons]
        self.hidden_neurons = [Neuron(id) for id in hidden_neurons]
        self.connections = [Connection(conn.from_id, conn.to_id, conn.weight) for conn in connections]

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

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
