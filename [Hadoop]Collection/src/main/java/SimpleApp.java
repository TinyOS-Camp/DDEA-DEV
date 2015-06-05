import com.fasterxml.jackson.databind.ObjectMapper;
import com.kmeg_os.ddea.collector.SDHMetaRetrieval;
import com.kmeg_os.ddea.collector.SDHReadingRetrieval;
import com.kmeg_os.ddea.model.sdh.SDHMetaItem;
import com.kmeg_os.ddea.model.sdh.SDHReading;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SimpleApp
{
    static Configuration hadoopConf;
    static FileSystem hadoopFs;
    static final int N_PROC = 8;

    public static void main(String[] args) {

        hadoopConf = new Configuration();
        hadoopConf.setBoolean("io.native.lib.available", true);

        try {
            hadoopFs = FileSystem.get(hadoopConf);

        } catch (Exception e) {
            e.printStackTrace();
        }


        boolean flag = Boolean.getBoolean(hadoopFs.getConf().get("dfs.support.append"));
        System.out.println("dfs.support.append is set to be " + flag);


        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        final Calendar initCal = Calendar.getInstance();
        initCal.roll(Calendar.YEAR, true);
        initCal.roll(Calendar.MONTH, true);
        initCal.roll(Calendar.DATE, true);

        initCal.set(Calendar.YEAR, 2011);
        initCal.set(Calendar.MONTH, 5);
        initCal.set(Calendar.DAY_OF_MONTH, 1);
        initCal.set(Calendar.HOUR_OF_DAY, 0);
        initCal.set(Calendar.MINUTE, 0);
        initCal.set(Calendar.SECOND, 0);
        System.out.println(dateFormat.format(initCal.getTime()));

        Date s = initCal.getTime();
        initCal.add(Calendar.DAY_OF_MONTH, 87 * 7);
        Date e = initCal.getTime();

        System.out.println("Start : " + dateFormat.format(s) + " End : " + dateFormat.format(e));
        collectMetaData();
        collectSDHReading(initCal, 90);
    }


    public static void collectMetaData(){


        try{

            SDHMetaRetrieval.collect(new SDHMetaRetrieval.SDHMetaItemResult() {
                @Override
                public void metaItemResult(SDHMetaItem[] result) {

                    try{

                        Path path = new org.apache.hadoop.fs.Path("/user/stkim1/SDH_READING_META.seq");
                        SequenceFile.Writer writer = null;
                        if(hadoopFs.exists(path)){
                            FSDataOutputStream out = hadoopFs.append(path);
                            writer = SequenceFile.createWriter(hadoopConf,out,Text.class,Text.class,SequenceFile.CompressionType.NONE,(CompressionCodec)null);
                        }else{
                            writer = SequenceFile.createWriter(hadoopFs, hadoopConf, path, Text.class, Text.class, SequenceFile.CompressionType.NONE, (CompressionCodec)null);
                        }

                        Text k = new Text();
                        Text v = new Text();

                        for (int i = 0; i < result.length; i++) {
                            final SDHMetaItem item = result[i];

                            k.set(item.uuid);
                            v.set(SDHMetaItem.formalizedParquetPath(item));
                            writer.append(k,v);
                        }

                        writer.close();
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
            });

        }catch(Exception e){
            e.printStackTrace();
        }
    }


    public static void collectSDHReading(final Calendar initCal, final int weekDuration){

        try {


            SDHMetaRetrieval.collect(new SDHMetaRetrieval.SDHMetaItemResult() {
                @Override
                public void metaItemResult(SDHMetaItem[] result) {

                    //ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                    ExecutorService executor = Executors.newFixedThreadPool(N_PROC);
                    try {

                        for (int i = 0; i < result.length; i++) {
                            final SDHMetaItem item = result[i];

                            executor.execute(new Runnable() {
                                @Override
                                public void run() {
                                    Path path = new org.apache.hadoop.fs.Path("/user/stkim1/" + SDHMetaItem.formalizedParquetPath(item) + ".seq");
                                    SequenceFile.Writer writer = null;

                                    try{
                                        if(hadoopFs.exists(path)){
                                            FSDataOutputStream out = hadoopFs.append(path);
                                            writer = SequenceFile.createWriter(hadoopConf,out,LongWritable.class, DoubleWritable.class,SequenceFile.CompressionType.NONE,(CompressionCodec)null);
                                        }else{
                                            writer = SequenceFile.createWriter(hadoopFs, hadoopConf, path, LongWritable.class, DoubleWritable.class, SequenceFile.CompressionType.NONE, (CompressionCodec)null);
                                        }

                                        saveItemReading(item, writer, initCal, weekDuration);

                                        writer.close();
                                    }catch (Exception e){
                                        e.printStackTrace();
                                    }

                                }
                            });
                        }
                        // This will make the executor accept no new threads
                        // and finish all existing threads in the queue
                        executor.shutdown();
                    } catch (Exception e) {
                        // (Re-)Cancel if current thread also interrupted
                        executor.shutdownNow();
                        // Preserve interrupt status
                        Thread.currentThread().interrupt();
                    }
                }
            });

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void saveItemReading(final SDHMetaItem item, final SequenceFile.Writer writer, final Calendar init, int weekDuration){

        System.out.println("READING [" + item.uuid + "] THREAD: " + Thread.currentThread().getId());

        Calendar c = Calendar.getInstance();
        c.roll(Calendar.YEAR, true);
        c.roll(Calendar.MONTH, true);
        c.roll(Calendar.DATE, true);
        c.setTime(init.getTime());

        for(int i = 0; i < weekDuration; i++ ){
            Date s = c.getTime();
            c.add(Calendar.DAY_OF_MONTH,7);
            Date e = c.getTime();

            long start = ((s.getTime() / 1000L) * 1000L);
            long end   = ((e.getTime() / 1000L) * 1000L);
            System.out.println("START : " + start + " | END : " + end);

            try{
                SDHReadingRetrieval.collect(item, start,end, new SDHReadingRetrieval.SDHReadingResult() {

                    @Override
                    public void readingResult(String tableName, SDHReading result) {
/*

                        [Python]
                        1 2 3 4 5 6 7
                        S M T W T F S

                        timetuple() : time.struct_time((d.year, d.month, d.day, 0, 0, 0, d.weekday(), yday, -1)),
                                      yday = d.toordinal() - date(d.year, 1, 1).toordinal() + 1

                        local_dt, time_tup[5], time_tup[4], time_tup[3], time_tup[6], time_tup[2], time_tup[1]

                        time_tup[5] : 0
                        time_tup[4] : 0
                        time_tup[3] : 0
                        time_tup[6] : d.weekday()
                        time_tup[2] : d.day
                        time_tup[1] : d.month

                        [Java]
                        0 1 2 3 4 5 6
                        M T W T F S S
                        key : (ts, weekday, day, month, value)

*/
                        LongWritable k = new LongWritable();
                        DoubleWritable v = new DoubleWritable();

                        try {
                            final int length = result.Readings.length;
                            for ( int i = 0; i < length; i++ )
                            {
                                double[] r = result.Readings[i];
                                if(r[0] == 0.0 && r[1] == 0.0)
                                {
                                    System.out.println(result.uuid + " : (" +Integer.valueOf(i).toString() + ") last ts [" + k.get() + "] value [" + v.get() + "]");
                                    continue;
                                }

                                long key = (long)(r[0] * 0.001);
                                k.set(key);
                                v.set(r[1]);
                                writer.append(k, v);
                            }

                            if(k.get() == 0 && v.get() == 0.0)
                            {
                                System.out.println(result.uuid  + " : last ts [" + k.get() + "] value [" + v.get() + "]");
                            }

                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                });

            }catch (Exception err){
                err.printStackTrace();
                break;
            }
        }
    }

}
