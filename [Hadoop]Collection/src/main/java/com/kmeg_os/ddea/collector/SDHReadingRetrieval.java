package com.kmeg_os.ddea.collector;

import com.esotericsoftware.minlog.Log;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kmeg_os.ddea.model.sdh.SDHMetaItem;
import com.kmeg_os.ddea.model.sdh.SDHReading;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.*;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.*;
import io.netty.util.CharsetUtil;
import org.eclipse.jetty.http.HttpHeaderValues;

import java.net.URI;

/**
 * Created by almightykim on 11/14/14.
 */
public class SDHReadingRetrieval  {
    static final ObjectMapper jsonMapper = new ObjectMapper();
    static final String TAG = SDHReadingRetrieval.class.getSimpleName();

    static private String UUID_PREFIX = "http://new.openbms.org/backend/api/data/uuid/";
    static private String DATE_FORMAT = "yyyy-MM-dd hh:mm:ss";


    public interface SDHReadingResult {
        public void readingResult(String tableName, SDHReading result);
    }

    public static void collect(final SDHMetaItem item, long start, long end, final SDHReadingResult receiver) throws Exception {

        URI uri = new URI(formalizedUrl(item,start,end));

        String scheme = uri.getScheme() == null? "http" : uri.getScheme();
        String host = uri.getHost() == null? "127.0.0.1" : uri.getHost();
        int port = uri.getPort();

        if (port == -1) {
            if ("http".equalsIgnoreCase(scheme)) {
                port = 80;
            } else if ("https".equalsIgnoreCase(scheme)) {
                port = 443;
            }
        }

        final StringBuilder buff = new StringBuilder();

        // Configure the client.
        EventLoopGroup group = new NioEventLoopGroup();
        try {
            Bootstrap b = new Bootstrap();
            b.group(group)
                    .channel(NioSocketChannel.class)
                    .handler(new ChannelInitializer<SocketChannel>() {

                        @Override
                        public void initChannel(SocketChannel ch) {
                            ChannelPipeline p = ch.pipeline();

                            p.addLast(new HttpClientCodec());

                            // Remove the following line if you don't want automatic content decompression.
                            p.addLast(new HttpContentDecompressor());

                            // Uncomment the following line if you don't want to handle HttpContents.
                            //p.addLast(new HttpObjectAggregator(1048576));

                            p.addLast(new SimpleChannelInboundHandler<HttpObject>(){

                                @Override
                                public void channelRead0(ChannelHandlerContext ctx, HttpObject msg) {


                                    if (false && (msg instanceof HttpResponse)) {
                                        HttpResponse response = (HttpResponse) msg;


                                        Log.info("THREAD: [" + Thread.currentThread().getId() + "]") ;

                                        Log.info("STATUS: " + response.getStatus());
                                        Log.info("VERSION: " + response.getProtocolVersion());

                                        if (!response.headers().isEmpty()) {
                                            for (String name: response.headers().names()) {
                                                for (String value: response.headers().getAll(name)) {
                                                    Log.info("HEADER: " + name + " = " + value);
                                                }
                                            }
                                        }
                                    }

                                    if (msg instanceof HttpContent) {

                                        HttpContent content = (HttpContent) msg;
                                        String str = content.content().toString(CharsetUtil.UTF_8);
                                        buff.append(str);

                                        if (content instanceof LastHttpContent) {

                                            try{
                                                SDHReading[] response = jsonMapper.readValue(buff.toString().getBytes(), SDHReading[].class);
                                                if (receiver != null){
                                                    receiver.readingResult(SDHMetaItem.formalizedParquetPath(item),response[0]);
                                                }

                                            }catch (Exception e){
                                                e.printStackTrace();
                                            }
                                            ctx.close();
                                        }
                                    }
                                }
                            });
                        }
                    });

            // Make the connection attempt.
            Channel ch = b.connect(host, port).sync().channel();

            // Prepare the HTTP request.
            HttpRequest request = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, formalizedUrl(item,start,end));
            request.headers().set(org.eclipse.jetty.http.HttpHeaders.HOST, host);
            request.headers().set(org.eclipse.jetty.http.HttpHeaders.CONNECTION, HttpHeaderValues.CLOSE);
            //request.headers().set(HttpHeaders.ACCEPT_ENCODING, HttpHeaderValues.GZIP);

            // Send the HTTP request.
            ch.writeAndFlush(request);

            // Wait for the server to close the connection.
            ch.closeFuture().sync();
        } finally {
            // Shut down executor threads to exit.
            group.shutdownGracefully();
        }
    }

    public static String formalizedUrl(SDHMetaItem item, long start, long end){
        return UUID_PREFIX + item.uuid + "?starttime=" + Long.valueOf(start).toString() + "&endtime=" + Long.valueOf(end).toString();
    }


}
