from kafka import KafkaConsumer
consumer = KafkaConsumer('foobar', group_id='foobar_group', bootstrap_servers='192.168.7.130:31090,192.168.7.130:31091,192.168.7.130:31092')
for msg in consumer:
    # print ('[Receive data] ' + msg)
    print (msg)