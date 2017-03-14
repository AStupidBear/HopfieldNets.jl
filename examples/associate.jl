using HopfieldNets, Plots
include("letters.jl")

patterns = hcat(X, O)
n = size(patterns, 1)

net = DiscreteHopfieldNet(n)

train!(net, patterns)
settle!(net)

Xcorrupt = copy(X)
Xcorrupt[2:7] = 1

Ocorrupt = copy(O)
Ocorrupt[2:7] = -1

heatmap(reshape(X, 7, 6))
heatmap(reshape(Xcorrupt, 7, 6))
heatmap(reshape(associate!(net, Xcorrupt), 7, 6))

heatmap(reshape(O, 7, 6))
heatmap(reshape(Ocorrupt, 7, 6))
heatmap(reshape(associate!(net, Ocorrupt), 7, 6))

heatmap(reshape(F1, 7, 6))
heatmap(reshape(associate!(net, F1), 7, 6))

heatmap(reshape(F2, 7, 6))
heatmap(reshape(associate!(net, F2), 7, 6))
