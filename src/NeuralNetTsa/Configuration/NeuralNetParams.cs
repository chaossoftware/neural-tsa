﻿using System.Text;
using System.Globalization;
using ChaosSoft.NeuralNetwork.Activation;

namespace NeuralNetTsa.Configuration;

/// <summary>
/// Class Describing parameters of Neural network
/// </summary>
public sealed class NeuralNetParams
{
    public NeuralNetParams(int neurons, int dimensions, int errorsExponent, int trainings, 
        IActivationFunction actFunction)
    {
        Neurons = neurons;
        Dimensions = dimensions;
        ErrorsExponent = errorsExponent;
        Trainings = trainings;
        ActFunction = actFunction;
    }

    public NeuralNetParams(int neurons, int dimensions, int errorsExponent, int trainings,
        IActivationFunction actFunction, double eta, long epochInterval, int biasTerm, 
        int constantTerm, double maxPertrubation, double nudge, int pruning, double testingInterval)
    {
        Neurons = neurons;
        Dimensions = dimensions;
        ErrorsExponent = errorsExponent;
        Trainings = trainings;
        ActFunction = actFunction;

        Eta = eta;
        EpochInterval = epochInterval;
        BiasTerm = biasTerm;
        ConstantTerm = constantTerm;
        MaxPertrubation = maxPertrubation;
        Nudge = nudge;
        Pruning = pruning;
        TestingInterval = testingInterval;
    }

    //Learning rate
    public double Eta { get; } = 0.999;

    //Number of iterations
    public long EpochInterval { get; } = 1000000;

    //0 for bias term; otherwise 1
    public int BiasTerm { get; } = 1;

    //0 for constant term; otherwise 1
    public int ConstantTerm { get; } = 0;

    //Maximum perturbation
    public double MaxPertrubation { get; } = 2;

    //Amount to nudge the parameters back toward zero
    public double Nudge { get; } = 0.5;

    //Pruning level (0 = no pruning)
    public int Pruning { get; } = 0;

    //Interval for testing neural net results
    public double TestingInterval { get; } = 1e5;

    //Neurons count
    public int Neurons { get; }

    //System dimensions
    public int Dimensions { get; }

    //Exponent of errors
    public int ErrorsExponent { get; }

    //Number of successful trainings to complete calculation
    public int Trainings { get; }

    //Activation function
    public IActivationFunction ActFunction { get; }

    public string GetInfoShort() => 
        new StringBuilder()
        .AppendFormat("Neurons             : {0}\n", Neurons)
        .AppendFormat("Dimensions          : {0}\n", Dimensions)
        .AppendFormat("Activation Function : {0}\n", ActFunction.Name)
        .AppendFormat("Exponent of error   : {0}\n", ErrorsExponent)
        .AppendFormat("Trainings count     : {0}\n", Trainings)
        .ToString();

    public string GetInfoFull() =>
        new StringBuilder()
        .Append(GetInfoShort() + "\n")
        .AppendFormat(CultureInfo.InvariantCulture, "Epoch interval      : {0:0.#####e-0}\n", EpochInterval)
        .AppendFormat(CultureInfo.InvariantCulture, "Testing interval    : {0:0.#####e-0}\n", TestingInterval)
        .AppendFormat(CultureInfo.InvariantCulture, "Learning rate       : {0}\n", Eta)
        .AppendFormat(CultureInfo.InvariantCulture, "Max. perturbation   : {0}\n", MaxPertrubation)
        .AppendFormat("Bias term           : {0}\n", BiasTerm)
        .AppendFormat("Constant term       : {0}\n", ConstantTerm)
        .AppendFormat("Pruning level       : {0}\n", Pruning)
        .AppendFormat(CultureInfo.InvariantCulture, "Amt. to nudge to 0  : {0}", Nudge)
        .ToString();
}
