namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class InputNeuron : NudgeNeuron
{
    public InputNeuron(int capacity) : base(1, capacity)
    {
    }

    public InputNeuron(double nudge, int capacity) : base(nudge, 1, capacity)
    {
    }

    public override void Process()
    {
        foreach (PruneSynapse synapse in Outputs)
        {
            synapse.Signal = Inputs[0].Signal * synapse.Weight;
        }
    }
}