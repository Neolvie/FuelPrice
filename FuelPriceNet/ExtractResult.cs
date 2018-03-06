namespace FuelPriceNet
{
  public class ExtractResult
  {
    public string Text { get; set; }
    public float Confidence { get; set; }

    public ExtractResult(string text, float confidence)
    {
      Text = text;
      Confidence = confidence;
    }
  }
}