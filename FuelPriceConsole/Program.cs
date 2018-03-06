using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FuelPriceConsole
{
  class Program
  {
    static void Main(string[] args)
    {
      var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "images");

      foreach (var file in Directory.GetFiles(path))
      {
        using (var image = new Bitmap(file))
        {
          var start = Environment.TickCount;
          var result = FuelPriceNet.ImageProcessor.GetText(image);
          Console.WriteLine($"-----------------{Path.GetFileName(file)}------------------");
          foreach (var extractResult in result)
          {
            var text = $"{extractResult.Text}, confidence {extractResult.Confidence}";
            Console.WriteLine(text);
          }
          Console.WriteLine($"Elapsed time: {Environment.TickCount - start} ms");
          Console.WriteLine($"-----------------------------------------------------------");
        }
        
        Console.ReadKey();
      }
    }
  }
}
