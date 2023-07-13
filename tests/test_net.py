import modnn
from modnn import utils

config = utils.read_config_file('./config.txt')
genome = modnn.Genome(config)
print(genome)
net = modnn.NN(genome)
net.visualize_graph()

print(net.a, net.b, net.c, net.d)


net.activate([0,0,0])
net.weight_update()
print("modulated")

for n in net.input_neurons:
    print(n.id, ':', n.modulated)
print("---")

for n in net.output_neurons:
    print(n.id, ':', n.modulated)
print("---")

for n in net.normal_neurons:
    print(n.id, ':', n.modulated)
print("---")

for n in net.lv1_neurons:
    print(n.id, ':', n.modulated)
print("---")

for n in net.lv2_neurons:
    print(n.id, ':', n.modulated)
print("---")
net.activate([0,0,0])
net.weight_update()
print("===")

net.activate([0,0,0])
net.weight_update()
print("===")
net.activate([0,0,0])
net.weight_update()
print("===")

print("PASSED")