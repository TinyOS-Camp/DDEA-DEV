__all__ = ['ClientClusterProtocol', 'BroadcastServerFactory']

from multiprocessing import Process
from Queue import Empty
from autobahn.twisted.websocket import WebSocketServerFactory, \
                                       WebSocketServerProtocol
from twisted.internet import reactor
from . import GET_BLOCK_TIME_SLICE

class ClientClusterProtocol(WebSocketServerProtocol):

    def onOpen(self):
        BroadcastServerFactory.register(self)

    def onMessage(self, payload, isBinary):
        ## echo back message verbatim
        self.sendMessage(payload, isBinary)

    def connectionLost(self, reason):
        WebSocketServerProtocol.connectionLost(self, reason)
        BroadcastServerFactory.unregister(self)


class BroadcastServerFactory(WebSocketServerFactory):
    clients = []

    @staticmethod
    def register(client):
        if not client in BroadcastServerFactory.clients:
            BroadcastServerFactory.clients.append(client)

    @staticmethod
    def unregister(client):
        return
        if client in BroadcastServerFactory.clients:
            BroadcastServerFactory.clients.remove(client)

    def __init__(self, url,target_in_queue):
        WebSocketServerFactory.__init__(self, url, debug = False,\
                                        debugCodePaths = False)
        self.target_in_queue = target_in_queue
        self.callback()

    def preparedBroadcast(self, msg):
        if msg is None:
            return

        preparedMsg = self.prepareMessage(msg)
        for c in BroadcastServerFactory.clients:
            c.sendPreparedMessage(preparedMsg)

    def broadcast(self, msg):
        if msg is None:
            return

        for c in self.clients:
            c.sendMessage(msg.encode('utf8'))

    def callback(self):
        try:
            out_data = None
            try:
                out_data = \
                    self.target_in_queue.get(block=True,
                                             timeout=GET_BLOCK_TIME_SLICE)
            except Empty as e:
                pass
            finally:

                if out_data:
                    self.target_in_queue.task_done()
                    self.broadcast(out_data)

        except (KeyboardInterrupt, SystemExit):
            exit(0)
        finally:
            reactor.callLater(GET_BLOCK_TIME_SLICE*0.001, self.callback)