package com.kmeg_os.ddea.model.sdh;

/**
 * Created by almightykim on 11/14/14.
 */
public class SparkSDH {

    private double ts;
    private double value;

    public SparkSDH(double t, double v){
        this.ts = t;
        this.value = v;
    }

    public double getTs(){
        return this.ts;
    }

    public void setTs(double t){
        this.ts = t;
    }

    public double getValue(){
        return this.value;
    }

    public void setValue(double v){
        this.value = v;
    }

}
