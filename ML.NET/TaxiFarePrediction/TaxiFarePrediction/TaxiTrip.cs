﻿using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace TaxiFarePrediction
{
    public class TaxiTrip
    {
        [Column("0")]
        public string VendorId;

        [Column("1")]
        public string RateCode;

        [Column("2")]
        public float PassengerCount;

        [Column("3")]
        public float TripTime;

        [Column("4")]
        public float TripDistance;

        [Column("5")]
        public string PaymentType;

        [Column("6")]
        public float FareAmount;
    }

    public class TaxiTripFarePrediction
    {
        //In case of the regression task, the Score column contains predicted label values.
        [ColumnName("Score")]
        public float FareAmount;
    }
}
