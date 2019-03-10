﻿using System;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Threading;
using MathLib;
using MathLib.Data;
using MathLib.DrawEngine.Charts;
using NeuralAnalyser.Configuration;
using NeuralAnalyser.NeuralNet;

namespace NeuralAnalyser
{
    public class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

            var config = new Config();
            var data = new SourceData(config.File.FileName);

            data.SetTimeSeries(config.File.DataColumn - 1, 0, data.Length - 1, 1, false);

            if (!Directory.Exists(config.Output.OutDirectory))
            {
                Directory.CreateDirectory(config.Output.OutDirectory);
            }

            Logger.Init(config.Output.LogFileName);

            new SignalPlot(data.TimeSeries, config.Output.PlotsSize, 1)
                .Plot().Save(config.Output.SignalPlotFileName, ImageFormat.Png);

            new MapPlot(Ext.GeneratePseudoPoincareMapData(data.TimeSeries.YValues), config.Output.PlotsSize, 1)
                .Plot().Save(config.Output.PoincarePlotFileName, ImageFormat.Png);

            var neuralNetParameters = config.NeuralNet;
            var neuralNet = new SciNeuralNet(neuralNetParameters, data.TimeSeries.YValues);

            Logger.LogInfo(neuralNetParameters.GetInfoFull(), true);

            Console.Title = "Signal: " + config.Output.FileName + " | " + neuralNetParameters.ActFunction.Name;
            Console.WriteLine("\nStarting...");

            var calculations = new Calculations(neuralNetParameters, config.Output);

            neuralNet.CycleComplete += calculations.LogCycle;
            neuralNet.EpochComplete += calculations.PerformCalculations;

            neuralNet.Process();

            calculations.Visualizator.NeuralAnimation.SaveAnimation(config.Output.AnimationFileName);
        }
    }
}
