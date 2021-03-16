using System;
using System.Drawing;
using System.IO;

namespace NeuronBP
{
    public class Neuron
    {
        /// <summary>
        /// Количество картинок
        /// </summary>
        const int PicturesCount = 60000;
        /// <summary>
        /// Высота картинки
        /// </summary>
        const byte PictureHeight = 28;
        /// <summary>
        /// Ширина картинки
        /// </summary>
        const byte PictureWidth = 28;
        /// <summary>
        /// Количество входов сети
        /// </summary>
        const int InputCount = PictureHeight * PictureWidth;
        /// <summary>
        /// Количество выходов сети
        /// </summary>
        const byte OutputCount = 10;
        /// <summary>
        /// Параметр скорости обучения
        /// </summary>
        const double Alpha = 0.5;
        /// <summary>
        /// Максимально допустимое число итераций
        /// </summary>
        const int Nmax = 300;
        /// <summary>
        /// Количество нейронов на скрытом слое
        /// </summary>
        const byte HiddenNeuronCount = 20;
        /// <summary>
        /// Параметр точности обучения
        /// </summary>
        double Epsilon;
        

        int t = 0; //номер эпохи
        int error = 0;
        int error2 = 0;
        int limit = 0;

        /// <summary>
        /// Матрица весовых коэффициентов от входов к скрытому слою
        /// </summary>
        double[,] W = new double[InputCount, HiddenNeuronCount];
        /// <summary>
        /// Матрица весов, соединяющих скрытый и выходной слой
        /// </summary>
        double[,] V = new double[HiddenNeuronCount, OutputCount];

        int[] input = new int[PictureHeight * PictureWidth];    
   
        double delta;
        /// <summary>
        /// Полученное реальное значение k-го выхода нейросети
        /// </summary>
        int y; //фактический результат
        int yk; //ожидаемый результат
        int[,] trainImages = new int[PicturesCount, InputCount];
        int[] trainLabels = new int[PicturesCount];
        int[,] testImages = new int[10000, InputCount];
        int[] testLabels = new int[10000];

        //yk – полученное реальное значение k-го выхода нейросети при подаче
        //на нее одного из входных образов обучающей выборки; dk – требуемое(целевое)
        //значение k-го выхода для этого образа.

        public void Learn()
        {
            if (File.Exists(@"../../../Resources/weight.txt")) LoadWeight();
            else
            {
                //Считывание dataset
                int imgCount = 0;
                foreach (var image in MnistReader.ReadTrainingData())
                {
                    int k = 0;
                    for (int i = 0; i < PictureHeight; i++)
                    {
                        for (int j = 0; j < PictureWidth; j++)
                        {
                            trainImages[imgCount, k++] = image.Data[i, j] > 0? 1 : 0;
                        }
                    }
                    trainLabels[imgCount++] = image.Label;
                }
                imgCount = 0;
                foreach (var image in MnistReader.ReadTestData())
                {
                    int k = 0;
                    for (int i = 0; i < PictureHeight; i++)
                    {
                        for (int j = 0; j < PictureWidth; j++)
                        {
                            testImages[imgCount, k++] = image.Data[i, j] > 0 ? 1 : 0;
                        }
                    }
                    testLabels[imgCount++] = image.Label;
                }

                Random rand = new Random((int)DateTime.Now.Ticks);
                for (int n = 0; n < OutputCount; n++)
                {
                    for (int i = 0; i < PictureHeight * PictureWidth; i++)
                    {
                        W[n, i] = Convert.ToDouble(rand.Next(-3, 4) / 10.0);
                    }
                }
                MnistCheck();
            }
        }

        public void MnistCheck()
        {
            t += 1;
            error = 0;
            error2 = 0;
            double sum;

            for (int img = 0; img < PicturesCount; img++)
            {
                for (int n = 0; n < OutputCount; n++)
                {
                    sum = 0;
                    for (int i = 0; i < PictureHeight * PictureWidth; i++)
                    {
                        sum += trainImages[img, i] * W[n, i];                      
                    }
                    y = sum > limit ? 1 : 0;
                    yk = n == trainLabels[img] ? 1 : 0;
                    if (y != yk)
                    {
                        delta = yk - y;
                        for (int i = 0; i < PictureHeight * PictureWidth; i++)
                        {
                            if (trainImages[img, i] == 1) W[n, i] += Alpha * delta;
                        }
                        error++;
                    }
                }
            }

            for (int img = 0; img < 10000; img++)
            {
                for (int n = 0; n < OutputCount; n++)
                {
                    sum = 0;
                    for (int i = 0; i < PictureHeight * PictureWidth; i++)
                    {
                        sum += testImages[img, i] * W[n, i];
                    }
                    y = sum > limit ? 1 : 0;
                    yk = n == testLabels[img] ? 1 : 0;
                    if (y != yk)
                    {
                        error2++;
                    }
                }
            }
            Console.WriteLine($"t = {t}");
            Console.WriteLine($"error = {error}");
            Console.WriteLine($"error2 = {error2}");
            if (t == 20)
            {
                Console.WriteLine("esf");
            }
            if (error > 0) MnistCheck();
            SaveWeight();
        }

        public double Sum(int n, Bitmap img)
        {
            double sum = 0; //сумма
            int[,] imgArray = new int[img.Width, img.Height]; //матрица входов
            LeadArray(CutImage(img, new Point(img.Width, img.Height)), input);
            for (int i = 0; i < PictureHeight * PictureWidth; i++)
            {
                sum += input[i] * W[n, i];                             
            }
            return sum;
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
            int[,] res = new int[28, 28];
            for (var n = 0; n < res.GetLength(0); n++)
                for (var m = 0; m < res.GetLength(1); m++) res[n, m] = 0;

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
            StreamWriter sw = new StreamWriter(@"../../../Resources/weight.txt");
            for (int n = 0; n < 10; n++)
            {
                for (int i = 0; i < PictureHeight * PictureWidth; i++)
                {
                    sw.WriteLine(W[n, i].ToString());
                }
            }            
            sw.Close();
        }

        public void LoadWeight()
        {
            var sr = new StreamReader(@"../../../Resources/weight.txt");
            for (int n = 0; n < 10; n++)
            {
                for (int i = 0; i < PictureHeight * PictureWidth; i++)
                {
                    W[n, i] = Convert.ToDouble(sr.ReadLine());                    
                }
            }
            sr.Close();
        }
        public void Recognize(Bitmap img)
        {
            //for (int n = 0; n < 10; n++)
            //{
                //if (Sum(n, img) > limit)
                //{
                //    MessageBox.Show($"Это {n}");
                //    break;
                //}
                //else if (n == 9)
                //{
                //    MessageBox.Show($"Не знаю ( ╯°□°)╯ ┻━━┻");
                //}                
            //};
        }

    }
}
