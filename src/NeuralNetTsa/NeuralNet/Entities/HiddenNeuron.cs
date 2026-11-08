using ChaosSoft.NeuralNetwork.Activation;

namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class HiddenNeuron : NudgeNeuron
{
    public static IActivationFunction Function;

    public HiddenNeuron(int inputs, int capacity) : base(inputs, capacity)
    {
    }

    public HiddenNeuron(double nudge, int inputs, int capacity) : base(nudge, inputs, capacity)
    {
    }

    public PruneSynapse BiasInput { get; set; }

    public override void Process()
    {
        double arg = BiasInput.Weight;

        foreach (PruneSynapse synapse in Inputs)
        {
            arg += synapse.Signal;
        }

        double multiplier = Function.Phi(arg);

        foreach (PruneSynapse synapse in Outputs)
        {
            synapse.Signal = synapse.Weight * multiplier;
        }
    }
}
