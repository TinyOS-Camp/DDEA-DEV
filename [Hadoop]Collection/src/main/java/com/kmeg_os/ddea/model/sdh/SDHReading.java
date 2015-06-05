package com.kmeg_os.ddea.model.sdh;

/**
 * Created by almightykim on 11/14/14.
 */

/*
  [
    {
        "uuid": "a31363a8-77de-5dc8-91e7-bf15c3b49d5e",
        "Readings": [
            [
                1414108816000,
                3.4
            ],
            [
                1414108847000,
                3.4
            ],
            [
                1414108867000,
                3.4
            ],
            [
                1414108887000,
                3.4
            ],
            [
                1414108907000,
                3.4
            ]
        ]
    }
]
*/

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public class SDHReading {
    @JsonProperty("uuid")
    public String uuid;

    @JsonProperty("Readings")
    //public List<List<Double>> Readings;
    public double[][] Readings;

}
