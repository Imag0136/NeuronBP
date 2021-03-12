using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace NeuronBP
{
    public class Neuron
    {
        byte t = 0; //номер эпохи
        byte error = 0;
        int limit = 0;
        double[,,] weight = new double[10, 28, 28]; //матрица весовых коэффициентов
        int[] input = new int[28];
        double alpha = 0.4; //Скорость обучения
        double delta;
        int y; //фактический результат
        int yk; //ожидаемый результат
        int[][,] data = new int[60000][,];
        byte[] label = new byte[60000];
        //Bitmap img;

        public Neuron()
        {
            //Установление случайных весов
            Random rand = new Random((int)DateTime.Now.Ticks);
            for (int n = 0; n < 10; n++)
            {
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        weight[n, i, j] = Convert.ToDouble(rand.Next(-3, 4) / 10.0);
                    }
                }
            }            
        }

        public void Learn()
        {
            if (File.Exists(@"../../Resources/weight.txt")) LoadWeight();
            else
            {
                //t += 1;
                //error = 0;
                //for (int k = 0; k < 50; k++)
                //{
                //    //img = (Bitmap)Image.FromFile($"../../Resources/{k}.jpg");                    
                //    for (int n = 0; n < 10; n++)
                //    {
                //        y = Sum(n, img) > limit ? 1 : 0;
                //        yk = n == (k / 5) ? 1 : 0;
                //        if (y != yk)
                //        {
                //            delta = yk - y;
                //            for (int i = 0; i < 100; i++)
                //            {
                //                if (input[i] == 1) weight[n, i, i] += alpha * delta;
                //            }
                //            error++;
                //        }
                //    }
                //}                
                //Console.WriteLine($"t = {t}");
                //Console.WriteLine($"error = {error}");
                //if (error > 0) Learn();
                //SaveWeight();

                //Считывание dataset
                int imgCount = 0;
                foreach (var image in MnistReader.ReadTrainingData())
                {
                    for (int i = 0; i < 28; i++)
                    {
                        for (int j = 0; j < 28; j++)
                        {
                            data[imgCount][i, j] = image.Data[i, j] == 0 ? 0 : 1;
                        }
                    }
                    label[imgCount] = image.Label;
                    imgCount++;
                }

                MnistCheck();
            }
        }

        public void MnistCheck()
        {
            t += 1;
            error = 0;
            double sum;

            for (int img = 0; img < 60000; img++)
            {
                for (int n = 0; n < 10; n++)
                {
                    sum = 0;
                    for (int i = 0; i < 28; i++)
                    {
                        for (int j = 0; j < 28; j++)
                        {
                            sum += data[img][i, j] / 255 * weight[n, i, j];
                        }
                    }
                    y = sum > limit ? 1 : 0;
                    yk = n == label[img] ? 1 : 0;
                    if (y != yk)
                    {
                        delta = yk - y;
                        for (int i = 0; i < 28; i++)
                        {
                            for (int j = 0; j < 28; j++)
                            {
                                if (data[img][i, j] / 255 == 1) weight[n, i, j] += alpha * delta;
                            }
                        }
                        error++;
                    }
                }
            }
            Console.WriteLine($"t = {t}");
            Console.WriteLine($"error = {error}");
            if (error > 20) MnistCheck();
            SaveWeight();
        }

        public double Sum(int n, Bitmap img)
        {
            double sum = 0; //сумма
            int[,] imgArray = new int[img.Width, img.Height]; //матрица входов
            LeadArray(CutImage(img, new Point(img.Width, img.Height)), input);
            for (int i = 0; i < 100; i++)
            {
                sum += input[i] * weight[n, i, i];
            }
            return sum;
        }

        // Процедура обрезание рисунка по краям и преобразование в массив.
        public static int[,] CutImage(Bitmap b, Point max)
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
            int[,] res = new int[28, 28];
            byte k = 0;
            for (var n = 0; n < res.GetLength(0); n++)
                for (var m = 0; m < res.GetLength(1); m++) res[n, m] = 0;

            var pX = (double)res.GetLength(0) / (double)source.GetLength(0);
            var pY = (double)res.GetLength(1) / (double)source.GetLength(1);

            for (var n = 0; n < source.GetLength(0); n++)
                for (var m = 0; m < source.GetLength(1); m++)
                {
                    var posX = (int)(n * pX);
                    var posY = (int)(m * pY);
                    if (res[posX, posY] == 0) res[posX, posY] = source[n, m];
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
            StreamWriter sw = new StreamWriter(@"../../Resources/weight.txt");
            for (int n = 0; n < weight.GetLength(0); n++)
            {
                for (int i = 0; i < weight.GetLength(1); i++)
                {
                    for (int j = 0; j < weight.GetLength(2); j++)
                    {
                        sw.WriteLine(weight[n, i, j].ToString());
                    }
                }
            }
            sw.Close();
        }

        public void LoadWeight()
        {
            var sr = new StreamReader(@"../../Resources/weight.txt");
            for (int n = 0; n < weight.GetLength(0); n++)
            {
                for (int i = 0; i < weight.GetLength(1); i++)
                {
                    for (int j = 0; j < weight.GetLength(2); j++)
                    {
                        weight[n, i, i] = Convert.ToDouble(sr.ReadLine());
                    }
                }
            }
            sr.Close();
        }
        public void Recognize(Bitmap img)
        {
            for (int n = 0; n < 10; n++)
            {
                if (Sum(n, img) > limit)
                {
                    MessageBox.Show($"Это {n}");
                    break;
                }
                else if (n == 9)
                {
                    MessageBox.Show($"Не знаю ( ╯°□°)╯ ┻━━┻");
                }
            };
        }

    }
}
