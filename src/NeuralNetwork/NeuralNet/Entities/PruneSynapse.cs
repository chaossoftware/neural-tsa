﻿using NewMind.NeuralNet.Base.Entities;

namespace NeuralAnalyser.NeuralNet.Entities
{
    public class PruneSynapse : Synapse
    {
        public PruneSynapse(int sourceIndex, int destinationIndex) 
            : base(sourceIndex, destinationIndex)
        {
            Prune = false;
        }

        public PruneSynapse(int sourceIndex, int destinationIndex, double weight)
            : base(sourceIndex, destinationIndex, weight)
        {
            Prune = false;
        }

        public bool Prune { get; set; }

        public void PruneIfMarked()
        {
            if (Prune)
            {
                Weight = 0;
            }
        }

        public override object Clone()
        {
            var synapseCopy = new PruneSynapse(IndexSource, IndexDestination)
            {
                Weight = this.Weight,
                Signal = this.Signal,
                Prune = this.Prune
            };

            return synapseCopy;
        }
    }
}
