from kafka import KafkaConsumer
consumer = KafkaConsumer('foobar', group_id='foobar_group', bootstrap_servers=['localhost:31090']
                            , auto_offset_reset='earliest', enable_auto_commit=False)
for message in consumer:
    # print ('[Receive data] ' + msg)
    print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                          message.offset, message.key,
                                          message.value))