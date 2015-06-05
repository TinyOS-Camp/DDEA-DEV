package com.kmeg_os.ddea.collector; /**
 * Created by almightykim on 14. 11. 12.
 */

import com.esotericsoftware.minlog.Log;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kmeg_os.ddea.model.sdh.SDHMetaItem;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.*;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.*;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.util.CharsetUtil;
import org.eclipse.jetty.http.HttpHeaderValues;
import org.eclipse.jetty.http.HttpHeaders;

import java.net.URI;

public class SDHMetaRetrieval {

    static final ObjectMapper jsonMapper = new ObjectMapper();
    static final String TAG = SDHMetaRetrieval.class.getSimpleName();
    static final String URL = System.getProperty("url", "http://new.openbms.org/backend/api/tags/Metadata__SourceName/Soda%20Hall%20Dent%20Meters");

    public interface SDHMetaItemResult {
        public void metaItemResult(SDHMetaItem[] result);
    }

    public static void collect(final SDHMetaItemResult receiver) throws Exception {

        Log.info(TAG, " Start collecting ");

        URI uri = new URI(URL);
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

        if (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) {
            Log.error("Only HTTP(S) is supported.");
            return;
        }

        // Configure SSL context if necessary.
        final boolean ssl = "https".equalsIgnoreCase(scheme);
        final SslContext sslCtx;
        if (ssl) {
            sslCtx = SslContext.newClientContext(InsecureTrustManagerFactory.INSTANCE);
        } else {
            sslCtx = null;
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

                            // Enable HTTPS if necessary.
                            if (sslCtx != null) {
                                p.addLast(sslCtx.newHandler(ch.alloc()));
                            }

                            p.addLast(new HttpClientCodec());

                            // Remove the following line if you don't want automatic content decompression.
                            p.addLast(new HttpContentDecompressor());

                            // Uncomment the following line if you don't want to handle HttpContents.
                            //p.addLast(new HttpObjectAggregator(1048576));

                            p.addLast(new SimpleChannelInboundHandler<HttpObject>(){

                                @Override
                                public void channelRead0(ChannelHandlerContext ctx, HttpObject msg) {
                                    if (msg instanceof HttpResponse) {
                                        HttpResponse response = (HttpResponse) msg;

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
                                                SDHMetaItem[] response = jsonMapper.readValue(buff.toString().getBytes(), SDHMetaItem[].class);

                                                if (receiver != null){
                                                    receiver.metaItemResult(response);
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
            HttpRequest request = new DefaultFullHttpRequest(
                    HttpVersion.HTTP_1_1, HttpMethod.GET, uri.getRawPath());
            request.headers().set(HttpHeaders.HOST, host);
            request.headers().set(HttpHeaders.CONNECTION, HttpHeaderValues.CLOSE);
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

}
