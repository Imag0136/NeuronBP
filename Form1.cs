using System;
using System.Drawing;
using System.Windows.Forms;

namespace NeuronBP
{
    public partial class Form1 : Form
    {
        Point current = new Point();
        Point old = new Point();
        Pen p = new Pen(Color.Black, 5);
        Graphics g;
        Bitmap img;
        Neuron neuron = new Neuron();
        public int incr = 0;
        public Form1()
        {
            InitializeComponent();
            pictureBox1.Image = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            p.SetLineCap(System.Drawing.Drawing2D.LineCap.Round, System.Drawing.Drawing2D.LineCap.Round, System.Drawing.Drawing2D.DashCap.Round);

            neuron.Learn();
        }

        private void LoadWeight_Click(object sender, EventArgs e)
        {
            //OpenFileDialog ofd = new OpenFileDialog();
            //ofd.Filter = "Image Files(*.BMP;*.JPG;*.PNG)|*.BMP;*.JPG;*.PNG";
            //if (ofd.ShowDialog() == DialogResult.OK)
            //{
            //    img = new Bitmap(ofd.FileName);
            //    pictureBox1.Image = img;
            //}

            neuron.Train();
        }



        private void SaveButton_Click(object sender, EventArgs e)
        {
            if (pictureBox1.Image != null) //если в pictureBox есть изображение
            {
                DateTime now = DateTime.Now;
                pictureBox1.Image.Save($"{incr}.jpg");
                incr += 1;
            }
        }

        private void recognizeButton_Click_1(object sender, EventArgs e)
        {
            //label0.Text = $"0: {neuron.Recognize(0, img)}";
            //label1.Text = $"1: {neuron.Recognize(1, img)}";
            //label2.Text = $"2: {neuron.Recognize(2, img)}";
            //label3.Text = $"3: {neuron.Recognize(3, img)}";
            //label4.Text = $"4: {neuron.Recognize(4, img)}";
            //label5.Text = $"5: {neuron.Recognize(5, img)}";
            //label6.Text = $"6: {neuron.Recognize(6, img)}";
            //label7.Text = $"7: {neuron.Recognize(7, img)}";
            //label8.Text = $"8: {neuron.Recognize(8, img)}";
            //label9.Text = $"9: {neuron.Recognize(9, img)}";
            for (int i = 0; i < 10; i++)
            {
                if (neuron.Recognize(i, img) == 1) MessageBox.Show($"Это {i}");
            }
            pictureBox1.Image = new Bitmap(pictureBox1.Width, pictureBox1.Height);
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            pictureBox1.Image = new Bitmap(pictureBox1.Width, pictureBox1.Height);
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            old = e.Location;
        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                current = e.Location;
                img = (Bitmap)pictureBox1.Image;
                g = Graphics.FromImage(img);
                g.DrawLine(p, old, current);
                pictureBox1.Image = img;
                old = current;
            }
        }
    }
}
