package com.kmeg_os.ddea.model.sdh;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Created by almightykim on 11/12/14.
 */

/*

{
    "Properties": {
        "Timezone": "America/Los_Angeles",
        "UnitofMeasure": "PF",
        "ReadingType": "double"
    },
    "Path": "/m352/elt-B/ABC/apparent_pf",
    "uuid": "3c8b43ea-06f4-54dc-a02c-fcac4ffbfd7f",
    "Metadata": {
        "SourceName": "Soda Hall Dent Meters",
        "Instrument": {
            "Model": "PowerScout 18",
            "SamplingPeriod": "20",
            "Manufacturer": "Dent Industries"
        },
        "Location": {
            "Building": "Soda Hall",
            "Campus": "UCB"
        },
        "Extra": {
            "ServiceArea": "Room 362",
            "SystemType": "Electrical",
            "DentElement": "elt-B",
            "ServiceDetail": "AC-#36/cr-2 Room 340",
            "Driver": "smap.drivers.dent.Dent18",
            "System": "hvac",
            "CircuitVolts": "480",
            "MeterName": "Soda 352",
            "Phase": "ABC",
            "SystemDetail": "AC-#36/cr-2"
        }
    }
}
 */


public class SDHMetaItem{

    static private String PREFIX = "SDH";
    static private String PARQUET = ".parquet";

    public class Properties{

        @JsonProperty("Timezone")
        public String Timezone;

        @JsonProperty("UnitofMeasure")
        public String UnitofMeasure;

        @JsonProperty("ReadingType")
        public String ReadingType;
    }



    public class Metadata{

        public class Instrument{

            @JsonProperty("Model")
            public String Model;

            @JsonProperty("SamplingPeriod")
            public String SamplingPeriod;

            @JsonProperty("Manufacturer")
            public String Manufacturer;
        }

        public class Location{

            @JsonProperty("Building")
            public String Building;

            @JsonProperty("Campus")
            public String Campus;
        }


        public class Extra{

            @JsonProperty("ServiceArea")
            public String ServiceArea;

            @JsonProperty("SystemType")
            public String SystemType;

            @JsonProperty("DentElement")
            public String DentElement;

            @JsonProperty("ServiceDetail")
            public String ServiceDetail;

            @JsonProperty("Driver")
            public String Driver;

            @JsonProperty("System")
            public String System;

            @JsonProperty("CircuitVolts")
            public String CircuitVolts;

            @JsonProperty("MeterName")
            public String MeterName;

            @JsonProperty("Phase")
            public String Phase;

            @JsonProperty("SystemDetail")
            public String SystemDetail;
        }

        @JsonProperty("SourceName")
        public String SourceName;

        @JsonProperty("Instrument")
        public Instrument Instrument;

        @JsonProperty("Location")
        public Location Location;

        @JsonProperty("Extra")
        public Extra Extra;

    }

    @JsonProperty("Properties")
    public Properties Properties;

    @JsonProperty("Path")
    public String Path;

    @JsonProperty("uuid")
    public String uuid;

    @JsonProperty("Metadata")
    public Metadata Metadata;

    public static String formalizedParquetPath(SDHMetaItem item){
        return PREFIX + item.Path.replaceAll("[\\/\\- ]","_").toUpperCase();// + PARQUET;
    }


}
