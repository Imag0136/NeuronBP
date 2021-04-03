using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;

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
        const int hiddenNeuronCount = 80;

        /// <summary>
        /// Количество выходов сети p
        /// </summary>
        const byte outputCount = 10;

        /// <summary>
        /// Параметр скорости обучения
        /// </summary>
        const float alpha = 0.7f;

        /// <summary>
        /// Максимально допустимое число итераций
        /// </summary>
        const int nmax = 100;

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

        /// <summary>
        /// Выходной вектор скрытого слоя
        /// </summary>
        float[] yc = new float[hiddenNeuronCount];

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

        int[] trainImg = new int[trainPicturesCount];
        int[] testImg = new int[testPicturesCount];
        public Bitmap picture;

        float[] biasH = new float[hiddenNeuronCount];
        float[] biasO = new float[outputCount];

        float[] prevMoment1 = new float[hiddenNeuronCount];
        float[] prevMoment2 = new float[inputCount];

        public void Learn()
        {
            if (File.Exists(@"../../../Resources/W.txt") && File.Exists(@"../../../Resources/V.txt")) LoadWeight();
            else
            {
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

                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    biasH[j] = (float)(rand.Next(-3, 4) / 10.0);
                }
                for (int k = 0; k < outputCount; k++)
                {
                    biasO[k] = (float)(rand.Next(-3, 4) / 10.0);
                }
                for (int i = 0; i < trainPicturesCount; i++)
                {
                    trainImg[i] = i;
                }
                for (int i = 0; i < testPicturesCount; i++)
                {
                    testImg[i] = i;
                }
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
                int label = trainImg[img] / (trainPicturesCount / outputCount);
                for (int i = 0; i < outputCount; i++)
                {
                    d[i] = i == label ? 1 : 0;
                }
                NetworkOutput(picture);
                WeightCorrect();

                for (int k = 0; k < outputCount; k++)
                {
                    epsilonTrain += (float)Math.Pow(y[k] - d[k], 2);
                }
                epsilonTrain /= 2;

                if (epsilonTrain > 0.01)
                {
                    errorTrain++;
                }
            }
            for (int img = 0; img < testPicturesCount; img++)
            {
                epsilonTest = 0;
                picture = (Bitmap)Image.FromFile($"../../../Resources/test/{testImg[img]}.jpg");
                int label = testImg[img] / (testPicturesCount / outputCount);
                for (int i = 0; i < outputCount; i++)
                {
                    d[i] = i == label ? 1 : 0;
                }
                NetworkOutput(picture);

                for (int k = 0; k < outputCount; k++)
                {
                    epsilonTest += (float)Math.Pow(y[k] - d[k], 2);
                }
                epsilonTest /= 2;

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

            if (epsilonTest > 0.001 && epoch < nmax)
            {
                Train();
            }
            else
            {
                Debug.WriteLine("Обучился");
                SaveWeight();
            }
        }

        /// <summary>
        /// Высчитывает выход нейронной сети
        /// </summary>
        void NetworkOutput(Bitmap img)
        {
            float[] sum1 = new float[hiddenNeuronCount];
            float[] sum2 = new float[outputCount];
            LeadArray(CutImage(img, new Point(img.Width, img.Height)), input);

            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    if (input[i] != 0)
                    {
                        sum1[j] += input[i] * w[i, j];
                    }
                }
                yc[j] = SigmoidFunction(sum1[j] + biasH[j]);
            }

            for (int k = 0; k < outputCount; k++)
            {
                for (int j = 0; j < hiddenNeuronCount; j++)
                {
                    sum2[k] += yc[j] * v[j, k];
                }
                y[k] = SigmoidFunction(sum2[k] + biasO[k]);                
            }
        }

        /// <summary>
        /// Активационная функция
        /// </summary>
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
                    float momentum = delta * yc[j] * prevMoment1[j] > 0 ? 1.2f : 0.5f;
                    v[j, k] -= alpha * delta * yc[j] * momentum;
                    sum[j] += delta * v[j, k];
                    prevMoment1[j] = delta * yc[j];
                }
                biasO[k] -= alpha * delta;
            }
            for (int j = 0; j < hiddenNeuronCount; j++)
            {
                for (int i = 0; i < inputCount; i++)
                {
                    if (input[i] != 0)
                    {
                        float momentum = sum[j] * yc[j] * (1f - yc[j]) * input[i] * prevMoment2[i] > 0 ? 1.2f : 0.5f;
                        w[i, j] -= alpha * (sum[j] * yc[j] * (1f - yc[j]) * input[i]) * momentum;
                        prevMoment2[i] = sum[j] * yc[j] * (1f - yc[j]) * input[i];
                    }
                }
                biasH[j] -= alpha * (sum[j] * yc[j] * (1f - yc[j]));
            }
        }

        void ShuffleTrain()
        {
            Random rand = new Random();
            for (int i = 0; i < trainImg.Length; i++)
            {
                int rnd = rand.Next();
                int j = rnd % (i + 1);
                int tmp = trainImg[j];
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
                int tmp = testImg[j];
                testImg[j] = testImg[i];
                testImg[i] = tmp;
            }
        }

        public double Recognize(int n, Bitmap img)
        {
            double ymax = 0d;
            double ymin = 0d;
            NetworkOutput(img);
            for (int k = 0; k < outputCount; k++)
            {
                if (ymax < y[k]) ymax = y[k];
                if (ymin > y[k]) ymin = y[k];
            }
            return (y[n] - ymin) / (ymax - ymin);
        }

        /// <summary>
        /// Процедура обрезание рисунка по краям и преобразование в массив
        /// </summary>
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

        /// <summary>
        /// Пересчёт матрицы source в массив res, для приведения произвольного массива данных к массиву стандартных размеров
        /// </summary>
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
    }
}
