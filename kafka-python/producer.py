from kafka import KafkaProducer
import json, time

def on_send_success(record_metadata):
    print('success send message')
    print('topic: ' + record_metadata.topic)
    print('partition: ' + str(record_metadata.partition))
    print('offset: ' + str(record_metadata.offset))

def on_send_error(excp):
    print('fail send message')
    print('I am an errback', exc_info=excp)
    # handle exception

producer = KafkaProducer(bootstrap_servers=['localhost:31090']
                            # , value_serializer=lambda m: json.dumps(m).encode('ascii')
                        )
for _ in range(1):
    topic = 'foobar'
    data = {'toName':'홍길동', 'toPhoneNumber':'01012344321', 'message':'한번 더 테스트합니다.'}
    # producer.send(topic, json.dumps(data).encode('ascii'))
    producer.send(topic, b'raw_bytes').add_callback(on_send_success).add_errback(on_send_error)

while True:
    time.sleep(1)