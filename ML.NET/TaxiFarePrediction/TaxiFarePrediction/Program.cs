using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;

namespace TaxiFarePrediction
{
    //https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/taxi-fare
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            var model = await TrainAsync();
            Evaluate(model);
            Predict(model, TestTrips.Trip1);
        }

        private static void Predict(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model, TaxiTrip trip1)
        {
            var prediction = model.Predict(trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"RMS = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }

        private static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> TrainAsync()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer("VendorId",
                "RateCode",
                "PaymentType"),
                new ColumnConcatenator("Features",
                "VendorId",
                "RateCode",
                "PassengerCount",
                "TripDistance",
                "PaymentType"),
                new FastTreeRegressor()
            };

            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelpath);

            return model;
        }
    }
}
