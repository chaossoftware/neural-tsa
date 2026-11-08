using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;
using System;
using System.Linq;

namespace NeuralNetTsa.Visualization;

public sealed class ConsoleNetVisualizer
{
    public const int EndOfOutput = ReportOffset + 8;
    private const int ReportOffset = 17;
    private const int InfoBorderOffset = 47;
    private const int RightPartOffset = 49;
    private const int TopRightOffset = 7;

    private const string InputNeuron = ">";
    private const string Synapse = "==>";

    private const int MinOffsetY = 0;

    private readonly string _delimiterLong = new string('─', InfoBorderOffset - 2) + "┬" + new string('─', 22);
    private readonly string _delimiter = new string('─', InfoBorderOffset - 2) + "┘";

    private readonly int _inCount;
    private readonly int _hiddenCount;
    private readonly int _maxItemsCount;
    private readonly int _inOffsetY;
    private readonly int _hiddenOffsetY;
    private readonly int _outOffsetY;

    private readonly ChaosNeuralNet _neuralNet;

    public ConsoleNetVisualizer(ChaosNeuralNet neuralNet)
    {
        _neuralNet = neuralNet;

        _inCount = neuralNet.Params.Dimensions;
        _hiddenCount = neuralNet.Params.Neurons;
        _maxItemsCount = Math.Max(_inCount, _hiddenCount);

        _inOffsetY = _maxItemsCount / _inCount + MinOffsetY;
        _hiddenOffsetY = _maxItemsCount / _hiddenCount + MinOffsetY;
        _outOffsetY = _maxItemsCount / neuralNet.OutputLayer.Length / 2 + MinOffsetY;
    }

    public void ReportCycle(ChaosNeuralNet net)
    {
        double error = net.OutputLayer[0].ShortMemory[0];
        string currentIteration = net.current.ToString("#,#").PadLeft(9);
        string currentEpoch = net.successCount.ToString().PadLeft(2);

        Console.SetCursorPosition(RightPartOffset, 3);
        Console.WriteLine($"Epoch {currentEpoch}: {currentIteration}");
        Console.SetCursorPosition(RightPartOffset, 5);
        Console.WriteLine($"ε  =  {net.Epsilon:e}");
        Console.SetCursorPosition(RightPartOffset, 6);
        Console.WriteLine($"e  =  {error:e}");

        Visualize();

        Console.SetCursorPosition(0, EndOfOutput);
    }

    public void PrintNetParams(NeuralNetParams neuralNetParams)
    {
        Console.WriteLine(_delimiterLong);

        Console.Write("\x1b[37m");
        Console.WriteLine(neuralNetParams.GetInfoFull());
        Console.ResetColor();

        for (int i = 2; i < ReportOffset; i++)
        {
            Console.SetCursorPosition(InfoBorderOffset - 2, i);
            Console.Write("│");
        }

        Console.SetCursorPosition(0, ReportOffset);
        Console.WriteLine(_delimiter);
    }

    private void Visualize()
    {
        double maxSynapseValue = _neuralNet.InputLayer.SelectMany(n => n.Outputs).Max(s => s.Signal);
        double minSynapseValue = _neuralNet.InputLayer.SelectMany(n => n.Outputs).Min(s => s.Signal);

        double segment = (maxSynapseValue - minSynapseValue) / 3d;
        double lowSynapseTreshold = minSynapseValue + segment;
        double highSynapseTreshold = minSynapseValue + 2d * segment;

        int i = 1;

        foreach (InputNeuron inn in _neuralNet.InputLayer)
        {
            Console.SetCursorPosition(RightPartOffset, i * _inOffsetY + TopRightOffset);
            Console.Write($"{InputNeuron} ");
            DrawSynapse(inn.Outputs[0].Signal, lowSynapseTreshold, highSynapseTreshold);
            i++;
        }

        maxSynapseValue = _neuralNet.HiddenLayer.SelectMany(n => n.Outputs).Max(s => s.Signal);
        minSynapseValue = _neuralNet.HiddenLayer.SelectMany(n => n.Outputs).Min(s => s.Signal);

        segment = (maxSynapseValue - minSynapseValue) / 3d;
        lowSynapseTreshold = minSynapseValue + segment;
        highSynapseTreshold = minSynapseValue + 2d * segment;
        i = 1;

        foreach (HiddenNeuron hn in _neuralNet.HiddenLayer)
        {
            Console.SetCursorPosition(RightPartOffset + 7, i * _hiddenOffsetY + TopRightOffset);
            DrawNeuron(hn);
            DrawSynapse(hn.Outputs[0].Signal, lowSynapseTreshold, highSynapseTreshold);
            i++;
        }

        Console.SetCursorPosition(RightPartOffset + 14, _outOffsetY + TopRightOffset);
        DrawNeuron(_neuralNet.OutputLayer[0]);
        DrawSynapse(1, 1, 1);
    }

    private static void DrawNeuron(HiddenNeuron neuron)
    {
        var color = neuron.Outputs[0].Signal > 0 ? "\x1b[32m" : "\x1b[37m";

        Console.Write(color);
        Console.Write("■ ");
        Console.ResetColor();
    }

    private static void DrawNeuron(OutputNeuron neuron)
    {
        var color = neuron.Outputs[0].Signal > 0 ? "\x1b[32m" : "\x1b[37m";

        Console.Write(color);
        Console.Write("● ");
        Console.ResetColor();
    }

    private static void DrawSynapse(double current, double firstThird, double secondThird)
    {
        if (current < firstThird)
        {
            Console.Write("\x1b[32m");
        }
        else if (current > secondThird)
        {
            Console.Write("\x1b[33m");
        }
        else
        {
            Console.Write("\x1b[37m");
        }

        Console.Write(Synapse);
        Console.ResetColor();
    }
}
