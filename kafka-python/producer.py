from kafka import KafkaProducer
import json

def on_send_success(record_metadata):
    print('success send message')
    print('topic: ' + record_metadata.topic)
    print('partition: ' + str(record_metadata.partition))
    print('offset: ' + str(record_metadata.offset))

def on_send_error(excp):
    print('fail send message')
    print('I am an errback', exc_info=excp)
    # handle exception

producer = KafkaProducer(bootstrap_servers='192.168.7.130:31090,192.168.7.130:31091,192.168.7.130:31092'
                            # , value_serializer=lambda m: json.dumps(m).encode('ascii')
                        )
for _ in range(1):
    topic = 'alarm-sms'
    data = {'toName':'황호성', 'toPhoneNumber':'01086608024', 'message':'한번 더 테스트합니다.'}
    producer.send(topic, json.dumps(data).encode('ascii'))

while True:
    time.sleep(1)