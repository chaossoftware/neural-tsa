namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class OutputNeuron : NudgeNeuron
{

    public OutputNeuron(int inputs) : base(inputs, 1)
    {
    }

    public OutputNeuron(double nudge, int inputs) : base(nudge, inputs, 1)
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

        Outputs[0].Signal = arg;
    }
}
