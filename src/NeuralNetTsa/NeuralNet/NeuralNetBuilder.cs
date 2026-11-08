using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet.Entities;

namespace NeuralNetTsa.NeuralNet;

public static class NeuralNetBuilder
{
    public static ChaosNeuralNet BuildNeuralNet(NeuralNetParams netParams, double[] array)
    {
        ChaosNeuralNet neuralNet = new(netParams, array);

        // init input layer
        for (int i = 0; i < netParams.Dimensions; i++)
        {
            InputNeuron neuron = new(netParams.Nudge, netParams.Neurons);
            neuron.Inputs[0] = new PruneSynapse(i, i, 1);
            neuralNet.InputLayer[i] = neuron;
        }

        // init constant neuron
        neuralNet.NeuronConstant = new BiasNeuron(netParams.Nudge, 1, netParams.Neurons);

        // init hidden layer
        HiddenNeuron.Function = netParams.ActFunction;

        for (int i = 0; i < netParams.Neurons; i++)
        {
            HiddenNeuron neuron = new(netParams.Nudge, netParams.Dimensions, 1);
            neuralNet.HiddenLayer[i] = neuron;
        }

        // init bias neuron
        neuralNet.NeuronBias = new BiasNeuron(netParams.Nudge, 0, 1);

        // init output layer
        OutputNeuron outNeuron = new(netParams.Nudge, netParams.Neurons);
        outNeuron.Outputs[0] = new PruneSynapse(0, 0, 1);
        neuralNet.OutputLayer[0] = outNeuron;

        //Connect input and hidden layer neurons
        for (int i = 0; i < netParams.Dimensions; i++)
        {
            for (int j = 0; j < netParams.Neurons; j++)
            {
                PruneSynapse synapse = new(i, j);
                neuralNet.InputLayer[i].Outputs[j] = synapse;
                neuralNet.HiddenLayer[j].Inputs[i] = synapse;
            }
        }

        //Connect constant and hidden neurons bias inputs
        for (int i = 0; i < netParams.Neurons; i++)
        {
            PruneSynapse constantSynapse = new(netParams.Dimensions, i);
            neuralNet.NeuronConstant.Outputs[i] = constantSynapse;
            neuralNet.HiddenLayer[i].BiasInput = constantSynapse;
        }

        //Connect hidden and output layer neurons
        for (int i = 0; i < netParams.Neurons; i++)
        {
            PruneSynapse synapse = new(i, 0);
            neuralNet.HiddenLayer[i].Outputs[0] = synapse;
            neuralNet.OutputLayer[0].Inputs[i] = synapse;
        }

        //Connect bias output and output neuron bias inputs
        PruneSynapse biasSynapse = new(netParams.Neurons, 1);
        neuralNet.NeuronBias.Outputs[0] = biasSynapse;
        neuralNet.OutputLayer[0].BiasInput = biasSynapse;

        return neuralNet;
    }
}