using System;

namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class BiasNeuron : NudgeNeuron
{
    public BiasNeuron(int inputs, int capacity) : base(inputs, capacity)
    {
    }

    public BiasNeuron(double nudge, int inputs, int capacity) : base(nudge, inputs, capacity)
    {
    }

    public override void Process() =>
        throw new NotSupportedException("Bias neuron has no inputs, so not able to process something");
}
