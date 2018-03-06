using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net.Mime;
using System.Runtime.InteropServices;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.ML;
using Tesseract;
using Point = OpenCvSharp.Point;
using Rect = Tesseract.Rect;
using Size = OpenCvSharp.Size;

namespace FuelPriceNet
{
  public static class ImageProcessor
  {
    private static Bitmap Image { get; set; }

    public static List<ExtractResult> GetText(Bitmap image)
    {
      Image = image;
      var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "tessdata");
      var results = new List<ExtractResult>();
      //PrepareImage(image);

      var contours = FindCountour(Image);
      //return new List<ExtractResult>();
      if (!contours.Any())
      {
        return new List<ExtractResult>() { new ExtractResult(string.Empty, 100) };
      }

      Console.WriteLine(contours.Count);
      foreach (var contour in contours)
      {
        var boundingRect = Cv2.BoundingRect(contour);
        var rectImage = new Rectangle(boundingRect.X, boundingRect.Y, boundingRect.Width, boundingRect.Height);
        using (var part = Image.Clone(rectImage, PixelFormat.DontCare))
        {
          SaveImage(part, "extracted_part");
          using (var engine = new TesseractEngine(path, "rus", EngineMode.TesseractAndLstm))
          {           
            using (var page = engine.Process(part, PageSegMode.SingleBlock))
            {              
              var result = new ExtractResult(page.GetText(), page.GetMeanConfidence());
              if (result.Confidence > 0.5)
                results.Add(result);
            }
          }
        }
      }

      return results;
    }

    private static List<Point[]> FindCountour(Bitmap image)
    {
      var original = BitmapConverter.ToMat(image);
      var gray = image.ToGrayscaleMat();
      var src = new Mat();
      gray.CopyTo(src);
      var threshImage = new Mat();
      gray.CopyTo(threshImage);

      //for (var i = 30; i <= 170; i += 20)
      //{
      //Cv2.Threshold(gray, threshImage, i, 255, ThresholdTypes.BinaryInv);
      //var denoise = new Mat();

      //Cv2.BilateralFilter(gray, denoise, 11, 17, 17);

      //Cv2.Threshold(~gray, threshImage, 20, 255, ThresholdTypes.);
      //SaveImage(threshImage, "thresh");

      //SaveImage(threshImage, $"thresh_min_{i}");
      //}
      //return;


      //SaveImage(threshImage, "thresh");
      //for (var i = 10; i < 70; i += 10)
      //{
      //  for (var j = 70; j <= 210; j += 20)
      //  {
      //Cv2.Canny(threshImage, threshImage, 30, 200);
      //Cv2.Canny(threshImage, threshImage, 50, 150);
      //SaveImage(threshImage, $"canny");
      //}
      //}

      var newThresh = new Mat();
      
      Cv2.AdaptiveThreshold(threshImage, newThresh, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 201, 15);
      //SaveImage(newThresh, "newThresh");
      threshImage = newThresh;

      Point[][] contours;
      HierarchyIndex[] hierarchyIndexes;
      Cv2.FindContours(threshImage, out contours, out hierarchyIndexes, RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);

      if (contours.Length == 0)
        return new List<Point[]>();

      //var dst = new Mat(gray.Rows, gray.Cols, MatType.CV_8UC3, Scalar.All(0));

      var sorted = contours
        .OrderByDescending(x => Cv2.ContourArea(x))
        .Take(10)
        .Where(x =>
        {
          var boundingRect = Cv2.BoundingRect(x);
          return boundingRect.IsVerticalBlock(1.5);
        }).ToList();

      if (!sorted.Any())
      {
        sorted = contours
          .OrderByDescending(x => Cv2.ContourArea(x))
          .Take(10)
          .Where(x =>
          {
            var boundingRect = Cv2.BoundingRect(x);
            return boundingRect.IsVerticalBlock(1);
          }).ToList();
      }

      foreach (var contour in sorted)
      {
        var boundingRect = Cv2.BoundingRect(contour);

        Cv2.Rectangle(original,
          new Point(boundingRect.X, boundingRect.Y),
          new Point(boundingRect.X + boundingRect.Width, boundingRect.Y + boundingRect.Height),
          new Scalar(0, 0, 255), 2);

        //Cv2.Rectangle(dst,
        //  new Point(boundingRect.X, boundingRect.Y),
        //  new Point(boundingRect.X + boundingRect.Width, boundingRect.Y + boundingRect.Height),
        //  new Scalar(0, 0, 255), 3);
      }

      //while (contourIndex >= 0)
      //{
      //  var contour = contours[contourIndex];
      //  var boundingRect = Cv2.BoundingRect(contour);

      //  Cv2.Rectangle(gray, 
      //    new Point(boundingRect.X, boundingRect.Y), 
      //    new Point(boundingRect.X + boundingRect.Width, boundingRect.Y + boundingRect.Height),
      //    new Scalar(0, 0, 255), 2);

      //  Cv2.Rectangle(dst,
      //    new Point(boundingRect.X, boundingRect.Y),
      //    new Point(boundingRect.X + boundingRect.Width, boundingRect.Y + boundingRect.Height),
      //    new Scalar(0, 0, 255), 2);

      //  contourIndex = hierarchyIndexes[contourIndex].Next;
      //}

      src = ~src;
      src = src * 1.2;   
      
      src = src + new Scalar(15, 15, 15);

      //SaveImage(src, "src");
      SaveImage(original, "result");
     
      Image = src.ToBitmap();

      return sorted;
    }

    private static bool IsVerticalBlock(this OpenCvSharp.Rect rect, double ratio)
    {
      return rect.Height > rect.Width * ratio;
    }

    private static Bitmap PrepareImage(Bitmap image)
    {
      //Mat src = new Mat(@"D:\tesseract4\docs\tables\balans_1kv_2013_21.jpg", ImreadModes.GrayScale);
      //var src = Cv2.ImRead(@"D:\tesseract4\docs\tables\balans_1kv_2013_21.jpg");
      var gray = image.ToGrayscaleMat();

      var bw = new Mat();

      Cv2.AdaptiveThreshold(~gray, bw, 256, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 25, -2);

      var horizontal = bw.Clone();
      var vertical = bw.Clone();
      var scale = 15;

      var horizontalSize = horizontal.Cols / scale;
      var verticalSize = vertical.Rows / scale;
      var horizontalStructure = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(horizontalSize, 1));
      var verticalStructure = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(1, verticalSize));

      Cv2.Erode(horizontal, horizontal, horizontalStructure, new OpenCvSharp.Point(-1, -1));
      Cv2.Dilate(horizontal, horizontal, horizontalStructure, new OpenCvSharp.Point(-1, -1));

      Cv2.Erode(vertical, vertical, verticalStructure, new OpenCvSharp.Point(-1, -1));
      Cv2.Dilate(vertical, vertical, verticalStructure, new OpenCvSharp.Point(-1, -1));

      //SaveImage(vertical, "vertical");

      //Cv2.Canny(src, dst, 50, 200);

      //using (new Window(horizontal))
      //{
      //  Cv2.WaitKey();
      //}
      var mask = horizontal + vertical;
      //SaveImage(mask, "mask");
      var newMask = new Mat();

      Cv2.AdaptiveThreshold(~mask, newMask, 256, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 7, -2);
      //SaveImage(newMask, "newMask");

      newMask = mask + newMask;

      SaveImage(newMask, "hyperMask");

      //var withOutTable = gray + newMask;

      //SaveImage(withOutTable, "withOutTable");

      return BitmapConverter.ToBitmap(mask);
    }

    private static void SaveImage(Mat mat, string name)
    {
      var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "prepared_images");
      if (!Directory.Exists(path))
        Directory.CreateDirectory(path);

      path = Path.Combine(path, $"{Environment.TickCount}_{name}.bmp");

      using (var img = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat))
      {
        img.Save(path);
      }
    }

    private static void SaveImage(Bitmap image, string name)
    {
      var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "prepared_images");
      if (!Directory.Exists(path))
        Directory.CreateDirectory(path);

      path = Path.Combine(path, $"{Environment.TickCount}_{name}.bmp");

      image.Save(path);
    }

    private static Mat ToGrayscaleMat(this Bitmap image)
    {
      var src = BitmapConverter.ToMat(image);
      var gray = new Mat();

      if (src.Channels() == 3)
      {
        Cv2.CvtColor(src, gray, ColorConversionCodes.RGB2GRAY);
      }
      else
      {
        gray = src;
      }

      return gray;
    }
  }
}
