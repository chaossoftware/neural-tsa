using System;
using System.Diagnostics;
using ChaosSoft.Core.DataUtils;
using MersenneTwister;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet.Entities;
using ChaosSoft.NeuralNetwork.Networks;
using NeuralNetTsa.NeuralNet.CustomActivation;
using NeuralNetTsa.Visualization;

namespace NeuralNetTsa.NeuralNet;

public sealed class ChaosNeuralNet : INeuralNet
{
    //----- input data
    private long nmax;  //lines in file
    public double[] xdata;

    //----- pre-calculated constants
    private double tenPowNegativePruning;
    private double nmaxSubD_xmaxPowE;
    private int nMul_DSubCtAdd1_AddNAdd1;
    private double[] djFactors;

    private double ddw;

    //counters
    public int current, successCount;

    private int improved = 0;
    private int seed;

    private bool AdditionalNeuron;

    public ChaosNeuralNet(NeuralNetParams taskParams, double[] timeseries)
    {
        Params = taskParams;

        InputLayer = new InputNeuron [taskParams.Dimensions];
        HiddenLayer = new HiddenNeuron [taskParams.Neurons];
        OutputLayer = new OutputNeuron [1];

        AdditionalNeuron = Params.ActFunction is ComplexActivationFunction;

        NeuronRandomizer.Randomizer = Randoms.Create(Params.Seed, RandomType.FastestDouble);

        nmax = timeseries.Length;
        double xmax = Vector.MaxAbs(timeseries);

        // create array with data
        xdata = new double[nmax];
        Array.Copy(timeseries, xdata, nmax);

        tenPowNegativePruning = Math.Pow(10, -Params.Pruning);
        nmaxSubD_xmaxPowE = (nmax - Params.Dimensions) * Math.Pow(xmax, Params.ErrorsExponent);
        nMul_DSubCtAdd1_AddNAdd1 = Params.Neurons * (Params.Dimensions - Params.ConstantTerm + 1) + Params.Neurons + 1;

        // Precompute 1/2^(minD5DivD * j) for all j values to avoid Math.Pow in hot loop
        djFactors = new double[Params.Dimensions];

        for (int j = 0; j < Params.Dimensions; j++)
        {
            djFactors[j] = 1d / Math.Pow(2d, Math.Min(Params.Dimensions, 5d) / Params.Dimensions * j);
        }

        ddw = Params.MaxPertrubation;
    }

    public delegate void NeuralNetEvent(ChaosNeuralNet network);

    public event NeuralNetEvent CycleComplete;

    public event NeuralNetEvent EpochComplete;

    public NeuralNetParams Params { get; set; }

    public BiasNeuron NeuronBias { get; set; }

    public BiasNeuron NeuronConstant { get; set; }

    public InputNeuron[] InputLayer { get; }

    public HiddenNeuron[] HiddenLayer { get; }

    public OutputNeuron[] OutputLayer { get; }

    public double Epsilon => Math.Abs(ddw);

    Stopwatch timer = Stopwatch.StartNew();

    public void Process()
    {
        while (successCount < Params.Trainings)
        {
            if (OutputLayer[0].LongMemory[0] == 0)
            {
                OutputLayer[0].ShortMemory[0] = 1000;
            }
            else
            {
                OutputLayer[0].ShortMemory[0] = 10 * OutputLayer[0].LongMemory[0];

                // OLD-IMPL
                // ddw = Math.Min(Params.MaxPertrubation, Math.Sqrt(OutputLayer[0].LongMemory[0]));

                // adaptive "cooling"
                // double coolingFactor = Math.Max(0.05, 1.0 - ((double)successCount / Params.Trainings));
                double coolingFactor = 1.0 - ((double)successCount / Params.Trainings);
                ddw = Math.Min(Params.MaxPertrubation, Math.Sqrt(OutputLayer[0].LongMemory[0]) * coolingFactor);
            }

            #region "update memory with best results"

            //update memory with best results
            foreach (InputNeuron neuron in InputLayer)
            {
                neuron.LongMemoryToShort();
            }

            NeuronConstant.LongMemoryToShort();

            foreach (HiddenNeuron neuron in HiddenLayer)
            {
                neuron.LongMemoryToShort();
            }

            NeuronBias.LongMemoryToShort();

            // same for Activation function neuron if needed
            if (AdditionalNeuron)
            {
                (Params.ActFunction as ComplexActivationFunction).Neuron.LongMemoryToShort();

                if ((Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemory[0] == 0)
                {
                    (Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemory[0] = 1;
                }
            }

            #endregion

            for (current = 1; current <= Params.EpochInterval; current++)
            {
                if (Params.Pruning == 0)
                {
                    foreach (InputNeuron neuron in InputLayer)
                    {
                        foreach (PruneSynapse synapse in neuron.Outputs)
                        {
                            synapse.Prune = false;
                        }
                    }

                    foreach (PruneSynapse synapse in NeuronConstant.Outputs)
                    {
                        synapse.Prune = false;
                    }
                }

                int prunes = 0;

                for (int i = 0; i < Params.Neurons; i++)
                {
                    for (int j = 0; j < Params.Dimensions; j++)
                    {
                        if (InputLayer[j].Outputs[i].Weight == 0)
                        {
                            prunes++;
                        }
                    }

                    if (NeuronConstant.Outputs[i].Weight == 0 && Params.ConstantTerm == 0)
                    {
                        prunes++;
                    }
                }

                //Probability of changing a given parameter at each trial
                //1 / Sqrt(neurons * (dims - Task_Params.ConstantTerm + 1) + neurons + 1 - prunes)
                double pc = 1d / Math.Sqrt(nMul_DSubCtAdd1_AddNAdd1 - prunes);

                if (Params.BiasTerm == 0)
                {
                    NeuronBias.CalculateWeight(0, ddw);
                }
                else
                {
                    NeuronBias.Outputs[0].Weight = 0;
                }

                for (int i = 0; i < Params.Neurons; i++)
                {
                    HiddenLayer[i].CalculateWeight(0, ddw, 9 * pc);

                    if (Params.ConstantTerm == 0)
                    {
                        NeuronConstant.CalculateWeight(i, ddw, pc);

                        //This connection has been pruned
                        NeuronConstant.Outputs[i].PruneIfMarked();
                    }

                    for (int j = 0; j < Params.Dimensions; j++)
                    {
                        //Reduce neighborhood for large j by a factor of 1-32
                        double dj = djFactors[j];

                        InputLayer[j].CalculateWeight(i, ddw * dj, pc);

                        //This connection has been pruned
                        InputLayer[j].Outputs[i].PruneIfMarked();
                    }
                }

                // same for Activation function neuron if needed
                if (AdditionalNeuron)
                {
                    for (int j = 0; j < 7; j++)
                    {
                        (Params.ActFunction as ComplexActivationFunction).Neuron.CalculateWeight(j, ddw, pc);
                    }
                }

                double e1 = 0;

                #region calculate 'mean square' error in prediction of all points

                for (int k = Params.Dimensions; k < nmax; k++)
                {
                    // get inputs from signal data
                    for (int j = 0; j < Params.Dimensions; j++)
                    {
                        InputLayer[j].Inputs[0].Signal = xdata[k - j - 1];
                    }

                    foreach (InputNeuron neuron in InputLayer)
                    {
                        neuron.Process();
                    }

                    foreach (HiddenNeuron neuron in HiddenLayer)
                    {
                        neuron.Process();
                    }

                    foreach (OutputNeuron neuron in OutputLayer)
                    {
                        neuron.Process();
                    }

                    //Error in the prediction of the k-th data point

                    // OLD-IMPL
                    // double ex = Math.Abs(OutputLayer[0].Outputs[0].Signal - xdata[k]);
                    // e1 += Math.Pow(ex, Params.ErrorsExponent);


                    double predPoint = OutputLayer[0].Outputs[0].Signal;
                    double realPoint = xdata[k];

                    // Error of point
                    double e_position = Math.Pow(Math.Abs(predPoint - realPoint), Params.ErrorsExponent);

                    // Prediction error of 1st derivative
                    double predVelocity = predPoint - xdata[k - 1];
                    double realVelocity = realPoint - xdata[k - 1];
                    double e_velocity = Math.Pow(Math.Abs(predVelocity - realVelocity), Params.ErrorsExponent);

                    // Final loss with weight coefficient
                    e1 += e_position + 0.2 * e_velocity;
                }

                //"Mean-square" error (even for e&<>2)
                e1 = Math.Pow(e1 / nmaxSubD_xmaxPowE, 2d / Params.ErrorsExponent);

                #endregion

                if (e1 < OutputLayer[0].ShortMemory[0])
                {
                    #region memorize current weights and new error
                    improved++;
                    OutputLayer[0].ShortMemory[0] = e1;

                    foreach (InputNeuron neuron in InputLayer)
                    {
                        neuron.WeightsToShortMemory();
                    }

                    foreach (HiddenNeuron neuron in HiddenLayer)
                    {
                        neuron.WeightsToShortMemory();
                    }

                    NeuronBias.WeightsToShortMemory();
                    NeuronConstant.WeightsToShortMemory();

                    // same for Activation function neuron if needed
                    if (AdditionalNeuron)
                    {
                        (Params.ActFunction as ComplexActivationFunction).Neuron.WeightsToShortMemory();
                    }

                    #endregion
                }
                else if (ddw > 0 && improved == 0)
                {
                    ddw = -ddw;     //Try going in the opposite direction
                }
                //Reseed the random if the trial failed
                else
                {
                    seed = NeuronRandomizer.Randomizer.Next(int.MaxValue);
                    //seed = (int)(1 / Math.Sqrt(e1));

                    if (improved > 0)
                    {
                        ddw = Math.Min(Params.MaxPertrubation, (1 + improved / Params.TestingInterval) * Math.Abs(ddw));
                        improved = 0;
                    }
                    else
                    {
                        ddw = Params.Eta * Math.Abs(ddw);
                    }
                }

                NeuronRandomizer.Randomizer = Randoms.Create(seed, RandomType.FastestDouble);

                //Testing is costly - don't do it too often
                if (current % Params.TestingInterval != 0)
                {
                    continue;
                }

                try
                {
                    CycleComplete.Invoke(this);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error during logging neural network cycle:\n" + ex);
                }

                if (OutputLayer[0].ShortMemory[0] > OutputLayer[0].LongMemory[0] && OutputLayer[0].LongMemory[0] != 0)
                {
                    continue;
                }

                // same for Activation function neuron if needed
                if (AdditionalNeuron)
                {
                    (Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemoryToWeights();
                }

                #region Mark the weak connections for pruning

                if (Params.Pruning != 0)
                {
                    for (int i = 0; i < Params.Neurons; i++)
                    {
                        for (int j = 0; j < Params.Dimensions; j++)
                        {
                            double aBest = InputLayer[j].ShortMemory[i];
                            double bBest = HiddenLayer[i].ShortMemory[0];

                            if (aBest != 0 && Math.Abs(aBest * bBest) < tenPowNegativePruning)
                            {
                                InputLayer[j].Outputs[i].Prune = true;
                            }
                        }

                        if (NeuronConstant.ShortMemory[i] != 0 && Math.Abs(NeuronConstant.ShortMemory[i] * HiddenLayer[i].ShortMemory[0]) < tenPowNegativePruning)
                        {
                            NeuronConstant.Outputs[i].Prune = true;
                        }
                    }
                }

                #endregion
            }

            #region "Save best weights"

            //Save best weights
            foreach (InputNeuron neuron in InputLayer)
            {
                neuron.ShortMemoryToLong();
            }

            foreach (HiddenNeuron neuron in HiddenLayer)
            {
                neuron.ShortMemoryToLong();
            }

            NeuronBias.ShortMemoryToLong();
            NeuronConstant.ShortMemoryToLong();

            // same for Activation function neuron if needed
            if (AdditionalNeuron)
            {
                (Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemoryToLong();
                (Params.ActFunction as ComplexActivationFunction).Neuron.LongMemoryToWeights();
            }

            #endregion

            successCount++;

            try
            {
                Console.SetCursorPosition(0, 18);
                Console.WriteLine($"Epoch duration: {timer.ElapsedMilliseconds:#,#} ms\n");
                EpochComplete.Invoke(this);
            }
            catch (Exception ex)
            {
                Console.SetCursorPosition(0, ConsoleNetVisualizer.EndOfOutput);
                Console.WriteLine("Error during performing calculations:\n" + ex.Message);
            }

            timer.Restart();
        }
    }

    void INeuralNet.ConstructNetwork()
        => throw new NotImplementedException();
}
