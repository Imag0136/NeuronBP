using System.Collections.Generic;
using System.IO;

namespace NeuronBP
{
    public static class MnistReader
    {
        private const string TrainImages = @"../../../Resources/mnist/train-images.idx3-ubyte";
        private const string TrainLabels = @"../../../Resources/mnist/train-labels.idx1-ubyte";
        private const string TestImages = @"../../../Resources/mnist/t10k-images.idx3-ubyte";
        private const string TestLabels = @"../../../Resources/mnist/t10k-labels.idx1-ubyte";

        public static IEnumerable<Imageee> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Imageee> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Imageee> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Imageee()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                };
            }
        }
    }
}
