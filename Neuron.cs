using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace NeuronBP
{
    public class Neuron
    {
        /// <summary>
        /// Количество картинок обучающей выборки
        /// </summary>
        const int trainPicturesCount = 100;

        /// <summary>
        /// Количество картинок тестовой выборки
        /// </summary>
        const int testPicturesCount = 20;

        /// <summary>
        /// Высота картинки
        /// </summary>
        const byte pictureHeight = 10;
        
        /// <summary>
        /// Ширина картинки
        /// </summary>
        const byte pictureWidth = 10;
        
        /// <summary>
        /// Количество входов сети n
        /// </summary>
        const int inputCount = pictureHeight * pictureWidth;
        
        /// <summary>
        /// Количество нейронов на скрытом слое m
        /// </summary>
        const int hiddenNeuronCount = 20;
        
        /// <summary>
        /// Количество выходов сети p
        /// </summary>
        const byte outputCount = 10;
        
        /// <summary>
        /// Параметр скорости обучения
        /// </summary>
        const float alpha = 0.1f;
        
        /// <summary>
        /// Максимально допустимое число итераций
        /// </summary>
        const int nmax = 1000;        
        
        /// <summary>
        /// Параметр точности обучения на обучающей выборке
        /// </summary>
        float epsilonTrain = 0.0f;
        
        /// <summary>
        /// Параметр точности обучения на тестовой выборке
        /// </summary>
        float epsilonTest = 0.0f;
        
        /// <summary>
        /// Номер эпохи
        /// </summary>
        int epoch = 0;
        
        /// <summary>
        /// Ошибка на обучающей выборке
        /// </summary>
        int errorTrain = 0;
        
        /// <summary>
        /// Ошибка на тестовой выборке
        /// </summary>
        int errorTest = 0;
        
        /// <summary>
        /// Матрица весовых коэффициентов от входов к скрытому слою
        /// </summary>
        float[,] w = new float[inputCount, hiddenNeuronCount];
        
        /// <summary>
        /// Матрица весов, соединяющих скрытый и выходной слой
        /// </summary>
        float[,] v = new float[hiddenNeuronCount, outputCount];

        float[] hiddenAnswerVector = new float[hiddenNeuronCount];
        /// <summary>
        /// Полученное реальное значение k-го выхода нейросети
        /// </summary>
        
        float[] y = new float[outputCount];
        /// <summary>
        /// Требуемое(целевое) значение k-го выхода для этого образа
        /// </summary>
        
        int[] d = new int[outputCount];
        /// <summary>
        /// Входной вектор
        /// </summary>
        
        int[] input = new int[inputCount];

        int[,] trainImages = new int[trainPicturesCount, inputCount];
        int[] trainLabels = new int[trainPicturesCount];
        int[,] testImages = new int[testPicturesCount, inputCount];
        int[] testLabels = new int[testPicturesCount];

        string[] trainImg = new string[trainPicturesCount];
        string[] testImg = new string[testPicturesCount];
        public Bitmap picture;

        public void Learn()
        {
            if (File.Exists(@"../../../Resources/W.txt") && File.Exists(@"../../../Resources/V.txt")) LoadWeight();
            else
            {
                //Считывание dataset
                //int imgCount = 0;
                //foreach (var image in MnistReader.ReadTrainingData())
                //{
                //    int k = 0;
                //    for (int i = 0; i < pictureHeight; i++)
                //    {
                //        for (int j = 0; j < pictureWidth; j++)
                //        {
                //            trainImages[imgCount, k++] = image.Data[i, j] > 0? 1 : 0;
                //        }
                //    }
                //    trainLabels[imgCount++] = image.Label;
                //}
                //imgCount = 0;
                //foreach (var image in MnistReader.ReadTestData())
                //{
                //    int k = 0;
                //    for (int i = 0; i < pictureHeight; i++)
                //    {
                //        for (int j = 0; j < pictureWidth; j++)
                //        {
                //            testImages[imgCount, k++] = image.Data[i, j] > 0 ? 1 : 0;
                //        }
                //    }
                //    testLabels[imgCount++] = image.Label;
                //}

                //Инициализация случайных весов
                Random rand = new Random((int)DateTime.Now.Ticks);
                for (int i = 0; i < inputCount; i++)
                {
                    for (int j = 0; j < hiddenNeuronCount; j++)
                    {
                        w[i, j] = (float)(rand.Next(-3, 4) / 10.0);
                    }
                }
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    for (int k = 0; k < outputCount; k++)
                    {
                        v[j, k] = (float)(rand.Next(-3, 4) / 10.0);
                    }
                }

                trainImg = RandomImageForLearn(trainPicturesCount);
                testImg = RandomImageForLearn(testPicturesCount);

                Train();
            }
        }

        public void Train()
        {
            epoch++;
            errorTrain = 0;
            errorTest = 0;
            epsilonTrain = 0;
            epsilonTest = 0;
            ShuffleTrain();
            ShuffleTest();

            for (int img = 0; img < trainPicturesCount; img++)
            {
                epsilonTrain = 0;
                picture = (Bitmap)Image.FromFile($"../../../Resources/train/{trainImg[img]}.jpg");
                int label = int.Parse(trainImg[img]) / (trainPicturesCount / outputCount);
                for (int i = 0; i < outputCount; i++)
                {
                    d[i] = i == label ? 1 : 0;
                }
                Sum(picture);
                WeightCorrect();

                for (int k = 0; k < outputCount; k++)
                {
                    epsilonTrain += (float)Math.Pow(y[k] - d[k], 2);
                }

                epsilonTrain /= 10;

                if (epsilonTrain > 0.01)
                {
                    errorTrain++;
                }
            }

            for (int img = 0; img < testPicturesCount; img++)
            {
                epsilonTest = 0;
                picture = (Bitmap)Image.FromFile($"../../../Resources/test/{testImg[img]}.jpg");
                int label = int.Parse(testImg[img]) / (testPicturesCount / outputCount);
                for (int i = 0; i < outputCount; i++)
                {
                    d[i] = i == label ? 1 : 0;
                }
                Sum(picture);

                for (int k = 0; k < outputCount; k++)
                    epsilonTest += (float)Math.Pow(y[k] - d[k], 2);

                epsilonTest /= 10;

                if (epsilonTest > 0.01)
                {
                    errorTest++;
                }
            }

            Debug.WriteLine($"Эпоха = {epoch}");
            Debug.WriteLine($"error1 = {errorTrain}");
            Debug.WriteLine($"error2 = {errorTest}");
            Debug.WriteLine($"epsilon1 = {epsilonTrain}");
            Debug.WriteLine($"epsilon2 = {epsilonTest}");
            Debug.WriteLine("");

            if (epsilonTest < 0.001)
            {

            }
            if (epsilonTest > 0.0005 && epoch < nmax)
            {
                Train();
            }
            else
            {
                Debug.WriteLine("Обучился");
                //SaveWeight();
            }
        }

        void Sum(Bitmap img)
        {
            float[] HiddenLayerSum = new float[hiddenNeuronCount];
            float[] OutputLayerSum = new float[outputCount];
            LeadArray(CutImage(img, new Point(img.Width, img.Height)), input);

            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    if (input[i] != 0)
                    {
                        HiddenLayerSum[j] += input[i] * w[i, j];
                    }
                }
                hiddenAnswerVector[j] = SigmoidFunction(HiddenLayerSum[j]);
            }

            for (int k = 0; k < outputCount; k++)
            {
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    OutputLayerSum[k] += hiddenAnswerVector[j] * v[j, k];
                }
                y[k] = SigmoidFunction(OutputLayerSum[k]);
            }
        }

        /// <summary>
        /// Активационная функция - сигмоида
        /// </summary>
        /// <param name="x">Сумма весов нейрона</param>
        /// <returns></returns>
        static float SigmoidFunction(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
        }

        void WeightCorrect()
        {
            float[] sum = new float[hiddenNeuronCount];
            for (int k = 0; k < outputCount; k++)
            {
                float delta = (y[k] - d[k]) * y[k] * (1f - y[k]);
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    v[j, k] -= alpha * delta * hiddenAnswerVector[j];
                    sum[j] += delta * v[j, k];
                }
            }
            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int i = 0; i < inputCount; i++)
                {
                    if (input[i] != 0)
                    {
                        w[i, j] -= alpha * (sum[j] * hiddenAnswerVector[j] * (1f - hiddenAnswerVector[j]) * input[i]);
                    }
                }
            }
        }

        void ShuffleTrain()
        {
            Random rand = new Random();
            for (int i = 0; i < trainImg.Length; i++)
            {
                int rnd = rand.Next();
                int j = rnd % (i + 1);
                string tmp = trainImg[j];
                trainImg[j] = trainImg[i];
                trainImg[i] = tmp;
            }
        }

        void ShuffleTest()
        {
            Random rand = new Random();
            for (int i = 0; i < testImg.Length; i++)
            {
                int rnd = rand.Next();
                int j = rnd % (i + 1);
                string tmp = testImg[j];
                testImg[j] = testImg[i];
                testImg[i] = tmp;
            }
        }

        public string[] RandomImageForLearn(int size)
        {
            string[] randomList = new string[size];
            for (int i = 0; i < size; i++)
            {
                randomList[i] = $"{i}";
            }
            return randomList;
        }

        public double Recognize(int n, Bitmap img)
        {
            double ymax = 0d;
            double ymin = 0d;
            Sum(img);
            for (int k = 0; k < outputCount; k++)
            {
                if (ymax < y[k]) ymax = y[k];
                if (ymin > y[k]) ymin = y[k];
            }
            return (y[n] - ymin)/(ymax-ymin);
        }

        // Процедура обрезание рисунка по краям и преобразование в массив.
        public int[,] CutImage(Bitmap b, Point max)
        {
            var x1 = 0;
            var y1 = 0;
            var x2 = max.X;
            var y2 = max.Y;

            for (var y = 0; y < b.Height && y1 == 0; y++)
                for (var x = 0; x < b.Width && y1 == 0; x++)
                    if (b.GetPixel(x, y).ToArgb() != 0) y1 = y;
            for (var y = b.Height - 1; y >= 0 && y2 == max.Y; y--)
                for (var x = 0; x < b.Width && y2 == max.Y; x++)
                    if (b.GetPixel(x, y).ToArgb() != 0) y2 = y;
            for (var x = 0; x < b.Width && x1 == 0; x++)
                for (var y = 0; y < b.Height && x1 == 0; y++)
                    if (b.GetPixel(x, y).ToArgb() != 0) x1 = x;
            for (var x = b.Width - 1; x >= 0 && x2 == max.X; x--)
                for (var y = 0; y < b.Height && x2 == max.X; y++)
                    if (b.GetPixel(x, y).ToArgb() != 0) x2 = x;

            if (x1 == 0 && y1 == 0 && x2 == max.X && y2 == max.Y) return null;

            var size = x2 - x1 > y2 - y1 ? x2 - x1 + 1 : y2 - y1 + 1;
            var dx = y2 - y1 > x2 - x1 ? ((y2 - y1) - (x2 - x1)) / 2 : 0;
            var dy = y2 - y1 < x2 - x1 ? ((x2 - x1) - (y2 - y1)) / 2 : 0;

            var res = new int[size, size];
            for (var x = 0; x < res.GetLength(0); x++)
                for (var y = 0; y < res.GetLength(1); y++)
                {
                    var pX = x + x1 - dx;
                    var pY = y + y1 - dy;
                    if (pX < 0 || pX >= max.X || pY < 0 || pY >= max.Y)
                        res[x, y] = 0;
                    else
                        res[x, y] = b.GetPixel(x + x1 - dx, y + y1 - dy).ToArgb() == 0 ? 0 : 1;
                }
            return res;
        }

        // Пересчёт матрицы source в массив res, для приведения произвольного массива данных к массиву стандартных размеров.
        public void LeadArray(int[,] source, int[] ans)
        {
            int k = 0;
            int[,] res = new int[pictureHeight, pictureWidth];

            var pX = (double)res.GetLength(0) / (double)source.GetLength(0);
            var pY = (double)res.GetLength(1) / (double)source.GetLength(1);

            for (var n = 0; n < source.GetLength(0); n++)
            {
                for (var m = 0; m < source.GetLength(1); m++)
                {
                    var posX = (int)(n * pX);
                    var posY = (int)(m * pY);
                    if (res[posX, posY] == 0) res[posX, posY] = source[n, m];
                }
            }
            for (int i = 0; i < res.GetLength(0); i++)
            {
                for (int j = 0; j < res.GetLength(1); j++)
                {
                    ans[k++] = res[i, j];
                }
            }
        }

        public void SaveWeight()
        {
            StreamWriter wStreamWriter = new StreamWriter(@"../../../Resources/W.txt");
            StreamWriter vStreamWriter = new StreamWriter(@"../../../Resources/V.txt");
            for (int i = 0; i < inputCount; i++)
            {
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    wStreamWriter.WriteLine(w[i, j].ToString());
                }
            }
            wStreamWriter.Close();
            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int k = 0; k < outputCount; k++)
                {
                    vStreamWriter.WriteLine(v[j, k].ToString());
                }
            }
            vStreamWriter.Close();
        }

        public void LoadWeight()
        {
            var wStreamReader = new StreamReader(@"../../../Resources/W.txt");
            var vStreamReader = new StreamReader(@"../../../Resources/V.txt");
            for (int i = 0; i < inputCount; i++)
            {
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    w[i, j] = float.Parse(wStreamReader.ReadLine());
                }
            }
            wStreamReader.Close();
            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int k = 0; k < outputCount; k++)
                {
                    v[j, k] = float.Parse(vStreamReader.ReadLine());
                }
            }
            vStreamReader.Close();
        }

        void OutputTestMnist(int img)
        {
            for (int k = 0; k < outputCount; k++)
            {
                d[k] = k == testLabels[img] ? 1 : 0;
            }
            float[] HiddenLayerSum = new float[hiddenNeuronCount];
            float[] OutputLayerSum = new float[outputCount];
            Array.Clear(HiddenLayerSum, 0, HiddenLayerSum.Length);
            Array.Clear(OutputLayerSum, 0, OutputLayerSum.Length);

            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int i = 0; i < inputCount; i++)
                {
                    if (testImages[img, i] != 0)
                    {
                        HiddenLayerSum[j] += testImages[img, i] * w[i, j];
                    }
                }
            }

            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                hiddenAnswerVector[j] = SigmoidFunction(HiddenLayerSum[j]);
            }

            for (int k = 0; k < outputCount; k++)
            {
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    OutputLayerSum[k] += hiddenAnswerVector[j] * v[j, k];
                }
            }

            for (int k = 0; k < outputCount; k++)
            {
                y[k] = SigmoidFunction(OutputLayerSum[k]);
            }
        }

        void OutputTrainMnist(int img)
        {
            for (int k = 0; k < outputCount; k++)
            {
                d[k] = k == trainLabels[img] ? 1 : 0;
            }
            float[] HiddenLayerSum = new float[hiddenNeuronCount];
            float[] OutputLayerSum = new float[outputCount];
            Array.Clear(HiddenLayerSum, 0, HiddenLayerSum.Length);
            Array.Clear(OutputLayerSum, 0, OutputLayerSum.Length);


            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int i = 0; i < inputCount; i++)
                {
                    if (trainImages[img, i] != 0)
                    {
                        HiddenLayerSum[j] += trainImages[img, i] * w[i, j];
                    }
                }
            }

            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                hiddenAnswerVector[j] = SigmoidFunction(HiddenLayerSum[j]);
            }

            for (int k = 0; k < outputCount; k++)
            {
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    OutputLayerSum[k] += hiddenAnswerVector[j] * v[j, k];
                }
            }

            for (int k = 0; k < outputCount; k++)
            {
                y[k] = SigmoidFunction(OutputLayerSum[k]);
            }
        }

        void WeightCorrectMnist(int img)
        {
            float[] TempHiddenLayerSum = new float[hiddenNeuronCount];
            Array.Clear(TempHiddenLayerSum, 0, TempHiddenLayerSum.Length);

            for (int k = 0; k < outputCount; k++)
            {
                float delta = (y[k] - d[k]) * y[k] * (1f - y[k]);
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    v[j, k] -= alpha * delta * hiddenAnswerVector[j];
                    TempHiddenLayerSum[j] += delta * v[j, k];
                }
            }

            for (int j = 0; j < hiddenNeuronCount; j++) // входной слой
            {
                for (int i = 0; i < inputCount; i++)
                {
                    if (trainImages[img, i] != 0)
                    {
                        w[i, j] -= alpha * (TempHiddenLayerSum[j] * hiddenAnswerVector[j] * (1f - hiddenAnswerVector[j]) * trainImages[img, i]);
                    }
                }
            }
        }
    }
}
