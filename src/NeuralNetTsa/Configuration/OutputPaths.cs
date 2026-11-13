using System.IO;

namespace NeuralNetTsa.Configuration;

public sealed class OutputPaths
{
    public OutputPaths(string outDir, string suffix, DataFileParams dataFileParams)
    {
        FileName = Path.GetFileName(dataFileParams.FileName);
        string suffixString = string.IsNullOrWhiteSpace(suffix) ? "" : "-" + suffix;
        OutDirectory = Path.Combine(outDir, $"nn_{FileName}_col-{dataFileParams.DataColumn}{suffixString}");
    }

    public string FileName { get; }

    public string OutDirectory { get; }

    public string LogFile => Path.Combine(OutDirectory, "log.txt");

    public string SignalPlotFile => Path.Combine(OutDirectory, FileName + "_signal.png");

    public string DelayedCoordPlotFile => Path.Combine(OutDirectory, FileName + "_delayed-coordinates.png");

    public string PredictFile => Path.Combine(OutDirectory, FileName + ".predict");

    public string ReconstSignalPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

    public string PredictedSignalPlotFile => Path.Combine(OutDirectory, FileName + "_prediction.png");

    public string ReconstrDelayedCoordPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_delayed-coordinates.png");

    public string ModelFile => Path.Combine(OutDirectory, FileName + "_model.3da");

    public string WavFile => Path.Combine(OutDirectory, FileName + "_sound.wav");

    public string AnimationFile => Path.Combine(OutDirectory, FileName + "_training_anim.gif");

    public string OverviewFile => Path.Combine(OutDirectory, FileName + "_overview.png");
}
