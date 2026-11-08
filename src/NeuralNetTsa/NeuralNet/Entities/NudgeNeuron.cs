using System;
using System.Runtime.CompilerServices;

namespace NeuralNetTsa.NeuralNet.Entities;

public abstract class NudgeNeuron
{
    protected NudgeNeuron(int inputs, int capacity) : this(0d, inputs, capacity)
    {
    }

    protected NudgeNeuron(double nudge, int inputs, int capacity)
    {
        Inputs = new PruneSynapse[inputs];
        Outputs = new PruneSynapse[capacity];
        ShortMemory = new double[capacity];
        LongMemory = new double[capacity];
        Nudge = nudge;
    }

    public PruneSynapse[] Inputs { get; }

    public PruneSynapse[] Outputs { get; }

    public double[] ShortMemory { get; }

    public double[] LongMemory { get; }

    public double Nudge { get; }

    public virtual void CalculateWeight(int index, double pertrubation)
    {
        Outputs[index].Weight = ShortMemory[index] + pertrubation * (Gauss2() - Nudge * Math.Sign(ShortMemory[index]));
    }

    public virtual void CalculateWeight(int index, double pertrubation, double lowerThan)
    {
        Outputs[index].Weight = ShortMemory[index];

        if(NeuronRandomizer.Randomizer.NextDouble() < lowerThan)
        {
            Outputs[index].Weight += pertrubation * (Gauss2() - Nudge * Math.Sign(ShortMemory[index]));
        }
    }

    public void LongMemoryToShort()
    {
        for(int i = 0; i < Outputs.Length; i++)
        {
            ShortMemory[i] = LongMemory[i];
        }
    }

    public void WeightsToShortMemory()
    {
        for (int i = 0; i < Outputs.Length; i++)
        {
            ShortMemory[i] = Outputs[i].Weight;
        }
    }

    public void ShortMemoryToWeights()
    {
        for (int i = 0; i < Outputs.Length; i++)
        {
            Outputs[i].Weight = ShortMemory[i];
        }
    }

    public void LongMemoryToWeights()
    {
        for (int i = 0; i < Outputs.Length; i++)
        {
            Outputs[i].Weight = LongMemory[i];
        }
    }

    public void ShortMemoryToLong()
    {
        for (int i = 0; i < Outputs.Length; i++)
        {
            LongMemory[i] = ShortMemory[i];
        }
    }

    public abstract void Process();

    public virtual object Clone() =>
        throw new NotImplementedException();

    /// <summary>
    /// Returns the product of two normally (Gaussian) distributed random
    /// deviates with meanof zero and standard deviation of 1.0 (Optimized)
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Gauss2()
    {
        // 1. Optimization: Local ref copy for randomizer.
        var rng = NeuronRandomizer.Randomizer;

        double v1, v2, arg;
        do
        {
            // 2. Optimization: use FMA instruction (var * const - const)
            v1 = rng.NextDouble() * 2d - 1d;
            v2 = rng.NextDouble() * 2d - 1d;
            arg = v1 * v1 + v2 * v2;
        }
        while (arg >= 1d || arg == 0d);

        // 3. Optimization:
        double invArg = 1d / arg;

        return v1 * v2 * (-2d + Math.Log(arg) * invArg);
    }
}
