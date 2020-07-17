from multiprocessing import Event, Process, Queue
from kafka import KafkaProducer

class Producer(Process):
    bootstrap_servers = ['192.168.7.130:31090',
                         '192.168.7.130:31091',
                         '192.168.7.130:31092']

    def __init__(self, topic):
        self.topic = topic
        self.queue = Queue()
        Process.__init__(self)
        self.stop_event = Event()

    def __del__(self):
        self.stop()

    def stop(self):
        self.stop_event.set()

    def run(self):
        producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)
        while not self.stop_event.is_set():
            if self.queue.qsize() != 0:
                data = self.queue.get()
                producer.send(self.topic, data)