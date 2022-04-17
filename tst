Starting inference_fast_online_1 ... 
Starting rabbitmq                ... 
Recreating inference_nginx_proxy_1 ... 
Starting mongodb                   ... 
Starting inference_fast_batch_1    ... 
Starting inference_node_1          ... 
Starting mongodb                   ... done
Starting inference_fast_batch_1    ... done
Starting inference_fast_online_1   ... done
Recreating inference_nginx_proxy_1 ... done
Starting inference_node_1          ... done
Starting rabbitmq                  ... done
Attaching to mongodb, inference_fast_batch_1, inference_fast_online_1, inference_nginx_proxy_1, inference_node_1, rabbitmq
[36mfast_batch_1   |[0m INFO:     Started server process [1]
[36mfast_batch_1   |[0m INFO:     Waiting for application startup.
[36mfast_batch_1   |[0m INFO:     Application startup complete.
[32mnginx_proxy_1  |[0m /docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
[32mnginx_proxy_1  |[0m /docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
[33mfast_online_1  |[0m None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[36mfast_batch_1   |[0m INFO:     Uvicorn running on http://0.0.0.0:8101 (Press CTRL+C to quit)
[32mnginx_proxy_1  |[0m /docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65040 - "OPTIONS /infer_results HTTP/1.1" 200 OK
[35mnode_1         |[0m INFO: Accepting connections at http://localhost:8000
[32mnginx_proxy_1  |[0m 10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
[32mnginx_proxy_1  |[0m 10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65040 - "GET /infer_results HTTP/1.1" 404 Not Found
[32mnginx_proxy_1  |[0m /docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.937+00:00"},"s":"I",  "c":"CONTROL",  "id":23285,   "ctx":"-","msg":"Automatically disabling TLS 1.0, to force-enable TLS 1.0 specify --sslDisabledProtocols 'none'"}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.938+00:00"},"s":"I",  "c":"NETWORK",  "id":4915701, "ctx":"main","msg":"Initialized wire specification","attr":{"spec":{"incomingExternalClient":{"minWireVersion":0,"maxWireVersion":13},"incomingInternalClient":{"minWireVersion":0,"maxWireVersion":13},"outgoing":{"minWireVersion":0,"maxWireVersion":13},"isInternalClient":true}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.939+00:00"},"s":"W",  "c":"ASIO",     "id":22601,   "ctx":"main","msg":"No TransportLayer configured during NetworkInterface startup"}
[32mnginx_proxy_1  |[0m /docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.939+00:00"},"s":"I",  "c":"NETWORK",  "id":4648601, "ctx":"main","msg":"Implicit TCP FastOpen unavailable. If TCP FastOpen is required, set tcpFastOpenServer, tcpFastOpenClient, and tcpFastOpenQueueSize."}
[32mnginx_proxy_1  |[0m /docker-entrypoint.sh: Configuration complete; ready for start up
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"W",  "c":"ASIO",     "id":22601,   "ctx":"main","msg":"No TransportLayer configured during NetworkInterface startup"}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"REPL",     "id":5123008, "ctx":"main","msg":"Successfully registered PrimaryOnlyService","attr":{"service":"TenantMigrationDonorService","ns":"config.tenantMigrationDonors"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"REPL",     "id":5123008, "ctx":"main","msg":"Successfully registered PrimaryOnlyService","attr":{"service":"TenantMigrationRecipientService","ns":"config.tenantMigrationRecipients"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"CONTROL",  "id":5945603, "ctx":"main","msg":"Multi threading initialized"}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"CONTROL",  "id":4615611, "ctx":"initandlisten","msg":"MongoDB starting","attr":{"pid":1,"port":27017,"dbPath":"/data/db","architecture":"64-bit","host":"87a4b0043485"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"CONTROL",  "id":23403,   "ctx":"initandlisten","msg":"Build Info","attr":{"buildInfo":{"version":"5.0.7","gitVersion":"b977129dc70eed766cbee7e412d901ee213acbda","openSSLVersion":"OpenSSL 1.1.1f  31 Mar 2020","modules":[],"allocator":"tcmalloc","environment":{"distmod":"ubuntu2004","distarch":"x86_64","target_arch":"x86_64"}}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"CONTROL",  "id":51765,   "ctx":"initandlisten","msg":"Operating System","attr":{"os":{"name":"Ubuntu","version":"20.04"}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.940+00:00"},"s":"I",  "c":"CONTROL",  "id":21951,   "ctx":"initandlisten","msg":"Options set by command line","attr":{"options":{"net":{"bindIp":"*"}}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.942+00:00"},"s":"I",  "c":"STORAGE",  "id":22270,   "ctx":"initandlisten","msg":"Storage engine to use detected by data files","attr":{"dbpath":"/data/db","storageEngine":"wiredTiger"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.942+00:00"},"s":"I",  "c":"STORAGE",  "id":22297,   "ctx":"initandlisten","msg":"Using the XFS filesystem is strongly recommended with the WiredTiger storage engine. See http://dochub.mongodb.org/core/prodnotes-filesystem","tags":["startupWarnings"]}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:40.942+00:00"},"s":"I",  "c":"STORAGE",  "id":22315,   "ctx":"initandlisten","msg":"Opening WiredTiger","attr":{"config":"create,cache_size=385813M,session_max=33000,eviction=(threads_min=4,threads_max=4),config_base=false,statistics=(fast),log=(enabled=true,archive=true,path=journal,compressor=snappy),builtin_extension_config=(zstd=(compression_level=6)),file_manager=(close_idle_time=600,close_scan_interval=10,close_handle_minimum=250),statistics_log=(wait=0),verbose=[recovery_progress,checkpoint_progress,compact_progress],"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.427+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:427807][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_PROGRESS] Recovering log 22 through 23"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.477+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:477383][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_PROGRESS] Recovering log 23 through 23"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.553+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:553748][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_ALL] Main recovery loop: starting at 22/7040 to 23/256"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.652+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:652652][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_PROGRESS] Recovering log 22 through 23"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.709+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:709851][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_PROGRESS] Recovering log 23 through 23"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.759+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:759212][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_ALL] Set global recovery timestamp: (0, 0)"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.759+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:759263][1:0x7f7f63c83c80], txn-recover: [WT_VERB_RECOVERY_ALL] Set global oldest timestamp: (0, 0)"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.760+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"initandlisten","msg":"WiredTiger message","attr":{"message":"[1650159041:760985][1:0x7f7f63c83c80], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 1, snapshot max: 1 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.762+00:00"},"s":"I",  "c":"STORAGE",  "id":4795906, "ctx":"initandlisten","msg":"WiredTiger opened","attr":{"durationMillis":820}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.762+00:00"},"s":"I",  "c":"RECOVERY", "id":23987,   "ctx":"initandlisten","msg":"WiredTiger recoveryTimestamp","attr":{"recoveryTimestamp":{"$timestamp":{"t":0,"i":0}}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.763+00:00"},"s":"I",  "c":"STORAGE",  "id":4366408, "ctx":"initandlisten","msg":"No table logging settings modifications are required for existing WiredTiger tables","attr":{"loggingEnabled":true}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.765+00:00"},"s":"I",  "c":"STORAGE",  "id":22262,   "ctx":"initandlisten","msg":"Timestamp monitor starting"}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.766+00:00"},"s":"W",  "c":"CONTROL",  "id":22120,   "ctx":"initandlisten","msg":"Access control is not enabled for the database. Read and write access to data and configuration is unrestricted","tags":["startupWarnings"]}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.768+00:00"},"s":"W",  "c":"CONTROL",  "id":22167,   "ctx":"initandlisten","msg":"You are running on a NUMA machine. We suggest launching mongod like this to avoid performance problems: numactl --interleave=all mongod [other options]","tags":["startupWarnings"]}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.771+00:00"},"s":"I",  "c":"NETWORK",  "id":4915702, "ctx":"initandlisten","msg":"Updated wire specification","attr":{"oldSpec":{"incomingExternalClient":{"minWireVersion":0,"maxWireVersion":13},"incomingInternalClient":{"minWireVersion":0,"maxWireVersion":13},"outgoing":{"minWireVersion":0,"maxWireVersion":13},"isInternalClient":true},"newSpec":{"incomingExternalClient":{"minWireVersion":0,"maxWireVersion":13},"incomingInternalClient":{"minWireVersion":13,"maxWireVersion":13},"outgoing":{"minWireVersion":13,"maxWireVersion":13},"isInternalClient":true}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.772+00:00"},"s":"I",  "c":"STORAGE",  "id":5071100, "ctx":"initandlisten","msg":"Clearing temp directory"}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.772+00:00"},"s":"I",  "c":"CONTROL",  "id":20536,   "ctx":"initandlisten","msg":"Flow Control is enabled on this deployment"}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.773+00:00"},"s":"I",  "c":"FTDC",     "id":20625,   "ctx":"initandlisten","msg":"Initializing full-time diagnostic data capture","attr":{"dataDirectory":"/data/db/diagnostic.data"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.776+00:00"},"s":"I",  "c":"REPL",     "id":6015317, "ctx":"initandlisten","msg":"Setting new configuration state","attr":{"newState":"ConfigReplicationDisabled","oldState":"ConfigPreStart"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.777+00:00"},"s":"I",  "c":"NETWORK",  "id":23015,   "ctx":"listener","msg":"Listening on","attr":{"address":"/tmp/mongodb-27017.sock"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.777+00:00"},"s":"I",  "c":"NETWORK",  "id":23015,   "ctx":"listener","msg":"Listening on","attr":{"address":"0.0.0.0"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.777+00:00"},"s":"I",  "c":"NETWORK",  "id":23016,   "ctx":"listener","msg":"Waiting for connections","attr":{"port":27017,"ssl":"off"}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.972+00:00"},"s":"I",  "c":"NETWORK",  "id":22943,   "ctx":"listener","msg":"Connection accepted","attr":{"remote":"172.27.0.1:45614","uuid":"78eff35d-e11f-4cac-9594-923020aa745c","connectionId":1,"connectionCount":1}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.973+00:00"},"s":"I",  "c":"NETWORK",  "id":51800,   "ctx":"conn1","msg":"client metadata","attr":{"remote":"172.27.0.1:45614","client":"conn1","doc":{"driver":{"name":"PyMongo","version":"4.1.1"},"os":{"type":"Linux","name":"Linux","architecture":"x86_64","version":"4.15.0-166-generic"},"platform":"CPython 3.9.12.final.0"}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.975+00:00"},"s":"I",  "c":"NETWORK",  "id":22944,   "ctx":"conn1","msg":"Connection ended","attr":{"remote":"172.27.0.1:45614","uuid":"78eff35d-e11f-4cac-9594-923020aa745c","connectionId":1,"connectionCount":0}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.975+00:00"},"s":"I",  "c":"NETWORK",  "id":22943,   "ctx":"listener","msg":"Connection accepted","attr":{"remote":"172.27.0.1:45618","uuid":"83ba5118-821b-4154-9812-f86c65a219d3","connectionId":2,"connectionCount":1}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.975+00:00"},"s":"I",  "c":"NETWORK",  "id":51800,   "ctx":"conn2","msg":"client metadata","attr":{"remote":"172.27.0.1:45618","client":"conn2","doc":{"driver":{"name":"PyMongo","version":"4.1.1"},"os":{"type":"Linux","name":"Linux","architecture":"x86_64","version":"4.15.0-166-generic"},"platform":"CPython 3.9.12.final.0"}}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.980+00:00"},"s":"I",  "c":"NETWORK",  "id":22943,   "ctx":"listener","msg":"Connection accepted","attr":{"remote":"172.27.0.1:45622","uuid":"150e19c8-d0ca-4207-a1a7-15fdad44aceb","connectionId":3,"connectionCount":2}}
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:30:41.981+00:00"},"s":"I",  "c":"NETWORK",  "id":51800,   "ctx":"conn3","msg":"client metadata","attr":{"remote":"172.27.0.1:45622","client":"conn3","doc":{"driver":{"name":"PyMongo","version":"4.1.1"},"os":{"type":"Linux","name":"Linux","architecture":"x86_64","version":"4.15.0-166-generic"},"platform":"CPython 3.9.12.final.0"}}}
[33mfast_online_1  |[0m Traceback (most recent call last):
[33mfast_online_1  |[0m   File "/online/release/./fastapi_runner.py", line 25, in <module>
[33mfast_online_1  |[0m     api_dict = {k:v.get_fast_apis()[0] for k,v in models_map.items()}
[33mfast_online_1  |[0m   File "/online/release/./fastapi_runner.py", line 25, in <dictcomp>
[33mfast_online_1  |[0m     api_dict = {k:v.get_fast_apis()[0] for k,v in models_map.items()}
[33mfast_online_1  |[0m   File "/online/release/squad_proto.py", line 90, in get_fast_apis
[33mfast_online_1  |[0m     return [SquadApi(self)]
[33mfast_online_1  |[0m   File "/online/release/squad_proto.py", line 27, in __init__
[33mfast_online_1  |[0m     super().__init__(proto, 'squad')
[33mfast_online_1  |[0m   File "/online/release/../model_proto/general_fastapi.py", line 31, in __init__
[33mfast_online_1  |[0m     self.rabbit_queue = RabbitRunQueue(proto.name)
[33mfast_online_1  |[0m   File "/online/release/../model_proto/rabbit_run_queue.py", line 15, in __init__
[33mfast_online_1  |[0m     tx_connection = pika.BlockingConnection(params)
[33mfast_online_1  |[0m   File "/usr/local/lib/python3.9/site-packages/pika/adapters/blocking_connection.py", line 360, in __init__
[33mfast_online_1  |[0m     self._impl = self._create_connection(parameters, _impl_class)
[33mfast_online_1  |[0m   File "/usr/local/lib/python3.9/site-packages/pika/adapters/blocking_connection.py", line 451, in _create_connection
[33mfast_online_1  |[0m     raise self._reap_last_connection_workflow_error(error)
[33mfast_online_1  |[0m pika.exceptions.IncompatibleProtocolError: StreamLostError: ("Stream connection lost: ConnectionResetError(104, 'Connection reset by peer')",)
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.356209+00:00 [info] <0.228.0> Feature flags: list of feature flags found:
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365458+00:00 [info] <0.228.0> Feature flags:   [x] implicit_default_bindings
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365510+00:00 [info] <0.228.0> Feature flags:   [x] maintenance_mode_status
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365528+00:00 [info] <0.228.0> Feature flags:   [x] quorum_queue
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365547+00:00 [info] <0.228.0> Feature flags:   [x] stream_queue
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365569+00:00 [info] <0.228.0> Feature flags:   [x] user_limits
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365587+00:00 [info] <0.228.0> Feature flags:   [x] virtual_host_metadata
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.365603+00:00 [info] <0.228.0> Feature flags: feature flag states written to disk: yes
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.628280+00:00 [noti] <0.44.0> Application syslog exited with reason: stopped
[36;1mrabbitmq       |[0m 2022-04-17 01:30:46.628374+00:00 [noti] <0.228.0> Logging: switching to configured handler(s); following messages may not be visible in this log output
[36;1mrabbitmq       |[0m [38;5;87m2022-04-17 01:30:46.638245+00:00 [notice] <0.228.0> Logging: configured log handlers are now ACTIVE[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.434820+00:00 [info] <0.228.0> ra: starting system quorum_queues[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.434898+00:00 [info] <0.228.0> starting Ra system: quorum_queues in directory: /var/lib/rabbitmq/mnesia/rabbit@9698263dcd24/quorum/rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.435936+00:00 [info] <0.293.0> ra system 'quorum_queues' running pre init for 0 registered servers[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.436594+00:00 [info] <0.297.0> ra: meta data store initialised for system quorum_queues. 0 record(s) recovered[0m
[36;1mrabbitmq       |[0m [38;5;87m2022-04-17 01:30:48.436942+00:00 [notice] <0.309.0> WAL: ra_log_wal init, open tbls: ra_log_open_mem_tables, closed tbls: ra_log_closed_mem_tables[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.438333+00:00 [info] <0.228.0> ra: starting system coordination[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.438376+00:00 [info] <0.228.0> starting Ra system: coordination in directory: /var/lib/rabbitmq/mnesia/rabbit@9698263dcd24/coordination/rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.439047+00:00 [info] <0.331.0> ra system 'coordination' running pre init for 0 registered servers[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.439777+00:00 [info] <0.336.0> ra: meta data store initialised for system coordination. 0 record(s) recovered[0m
[36;1mrabbitmq       |[0m [38;5;87m2022-04-17 01:30:48.440026+00:00 [notice] <0.347.0> WAL: ra_coordination_log_wal init, open tbls: ra_coordination_log_open_mem_tables, closed tbls: ra_coordination_log_closed_mem_tables[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.441357+00:00 [info] <0.228.0> [0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.441357+00:00 [info] <0.228.0>  Starting RabbitMQ 3.9.15 on Erlang 24.3.3 [jit][0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.441357+00:00 [info] <0.228.0>  Copyright (c) 2007-2022 VMware, Inc. or its affiliates.[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.441357+00:00 [info] <0.228.0>  Licensed under the MPL 2.0. Website: https://rabbitmq.com[0m
[36;1mrabbitmq       |[0m 
[36;1mrabbitmq       |[0m   ##  ##      RabbitMQ 3.9.15
[36;1mrabbitmq       |[0m   ##  ##
[36;1mrabbitmq       |[0m   ##########  Copyright (c) 2007-2022 VMware, Inc. or its affiliates.
[36;1mrabbitmq       |[0m   ######  ##
[36;1mrabbitmq       |[0m   ##########  Licensed under the MPL 2.0. Website: https://rabbitmq.com
[36;1mrabbitmq       |[0m 
[36;1mrabbitmq       |[0m   Erlang:      24.3.3 [jit]
[36;1mrabbitmq       |[0m   TLS Library: OpenSSL - OpenSSL 1.1.1n  15 Mar 2022
[36;1mrabbitmq       |[0m 
[36;1mrabbitmq       |[0m   Doc guides:  https://rabbitmq.com/documentation.html
[36;1mrabbitmq       |[0m   Support:     https://rabbitmq.com/contact.html
[36;1mrabbitmq       |[0m   Tutorials:   https://rabbitmq.com/getstarted.html
[36;1mrabbitmq       |[0m   Monitoring:  https://rabbitmq.com/monitoring.html
[36;1mrabbitmq       |[0m 
[36;1mrabbitmq       |[0m   Logs: /var/log/rabbitmq/rabbit@9698263dcd24_upgrade.log
[36;1mrabbitmq       |[0m         <stdout>
[36;1mrabbitmq       |[0m 
[36;1mrabbitmq       |[0m   Config file(s): /etc/rabbitmq/conf.d/10-defaults.conf
[36;1mrabbitmq       |[0m 
[36;1mrabbitmq       |[0m   Starting broker...2022-04-17 01:30:48.442343+00:00 [info] <0.228.0> [0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>  node           : rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>  home dir       : /var/lib/rabbitmq[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>  config file(s) : /etc/rabbitmq/conf.d/10-defaults.conf[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>  cookie hash    : b9rfDURKBI2E0HMftcRaLQ==[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>  log(s)         : /var/log/rabbitmq/rabbit@9698263dcd24_upgrade.log[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>                 : <stdout>[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.442343+00:00 [info] <0.228.0>  database dir   : /var/lib/rabbitmq/mnesia/rabbit@9698263dcd24[0m
[33;1mfast_online_1  |[0m None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[33;1mfast_online_1  |[0m Traceback (most recent call last):
[33;1mfast_online_1  |[0m   File "/online/release/./fastapi_runner.py", line 25, in <module>
[33;1mfast_online_1  |[0m     api_dict = {k:v.get_fast_apis()[0] for k,v in models_map.items()}
[33;1mfast_online_1  |[0m   File "/online/release/./fastapi_runner.py", line 25, in <dictcomp>
[33;1mfast_online_1  |[0m     api_dict = {k:v.get_fast_apis()[0] for k,v in models_map.items()}
[33;1mfast_online_1  |[0m   File "/online/release/squad_proto.py", line 90, in get_fast_apis
[33;1mfast_online_1  |[0m     return [SquadApi(self)]
[33;1mfast_online_1  |[0m   File "/online/release/squad_proto.py", line 27, in __init__
[33;1mfast_online_1  |[0m     super().__init__(proto, 'squad')
[33;1mfast_online_1  |[0m   File "/online/release/../model_proto/general_fastapi.py", line 31, in __init__
[33;1mfast_online_1  |[0m     self.rabbit_queue = RabbitRunQueue(proto.name)
[33;1mfast_online_1  |[0m   File "/online/release/../model_proto/rabbit_run_queue.py", line 15, in __init__
[33;1mfast_online_1  |[0m     tx_connection = pika.BlockingConnection(params)
[33;1mfast_online_1  |[0m   File "/usr/local/lib/python3.9/site-packages/pika/adapters/blocking_connection.py", line 360, in __init__
[33;1mfast_online_1  |[0m     self._impl = self._create_connection(parameters, _impl_class)
[33;1mfast_online_1  |[0m   File "/usr/local/lib/python3.9/site-packages/pika/adapters/blocking_connection.py", line 451, in _create_connection
[33;1mfast_online_1  |[0m     raise self._reap_last_connection_workflow_error(error)
[33;1mfast_online_1  |[0m pika.exceptions.IncompatibleProtocolError: StreamLostError: ('Transport indicated EOF',)
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645710+00:00 [info] <0.228.0> Feature flags: list of feature flags found:[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645750+00:00 [info] <0.228.0> Feature flags:   [x] drop_unroutable_metric[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645777+00:00 [info] <0.228.0> Feature flags:   [x] empty_basic_get_metric[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645815+00:00 [info] <0.228.0> Feature flags:   [x] implicit_default_bindings[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645833+00:00 [info] <0.228.0> Feature flags:   [x] maintenance_mode_status[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645856+00:00 [info] <0.228.0> Feature flags:   [x] quorum_queue[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645898+00:00 [info] <0.228.0> Feature flags:   [x] stream_queue[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645939+00:00 [info] <0.228.0> Feature flags:   [x] user_limits[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645956+00:00 [info] <0.228.0> Feature flags:   [x] virtual_host_metadata[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.645972+00:00 [info] <0.228.0> Feature flags: feature flag states written to disk: yes[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.868743+00:00 [info] <0.228.0> Running boot step pre_boot defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.868821+00:00 [info] <0.228.0> Running boot step rabbit_global_counters defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.869167+00:00 [info] <0.228.0> Running boot step rabbit_osiris_metrics defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.869285+00:00 [info] <0.228.0> Running boot step rabbit_core_metrics defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.871889+00:00 [info] <0.228.0> Running boot step rabbit_alarm defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.875963+00:00 [info] <0.361.0> Memory high watermark set to 309060 MiB (324073347481 bytes) of 772651 MiB (810183368704 bytes) total[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.880312+00:00 [info] <0.363.0> Enabling free disk space monitoring[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.880350+00:00 [info] <0.363.0> Disk free limit set to 50MB[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.884600+00:00 [info] <0.228.0> Running boot step code_server_cache defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.884793+00:00 [info] <0.228.0> Running boot step file_handle_cache defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.884977+00:00 [info] <0.368.0> Limiting to approx 1048479 file handles (943629 sockets)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.885079+00:00 [info] <0.369.0> FHC read buffering: OFF[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.885111+00:00 [info] <0.369.0> FHC write buffering: ON[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.885378+00:00 [info] <0.228.0> Running boot step worker_pool defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.885412+00:00 [info] <0.353.0> Will use 96 processes for default worker pool[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.885452+00:00 [info] <0.353.0> Starting worker pool 'worker_pool' with 96 processes in it[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.888746+00:00 [info] <0.228.0> Running boot step database defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.890543+00:00 [info] <0.228.0> Waiting for Mnesia tables for 30000 ms, 9 retries left[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.890742+00:00 [info] <0.228.0> Successfully synced tables from a peer[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.890805+00:00 [info] <0.228.0> Waiting for Mnesia tables for 30000 ms, 9 retries left[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.890874+00:00 [info] <0.228.0> Successfully synced tables from a peer[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899586+00:00 [info] <0.228.0> Waiting for Mnesia tables for 30000 ms, 9 retries left[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899656+00:00 [info] <0.228.0> Successfully synced tables from a peer[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899679+00:00 [info] <0.228.0> Peer discovery backend rabbit_peer_discovery_classic_config does not support registration, skipping registration.[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899745+00:00 [info] <0.228.0> Running boot step database_sync defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899803+00:00 [info] <0.228.0> Running boot step feature_flags defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899874+00:00 [info] <0.228.0> Running boot step codec_correctness_check defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899898+00:00 [info] <0.228.0> Running boot step external_infrastructure defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899921+00:00 [info] <0.228.0> Running boot step rabbit_registry defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.899971+00:00 [info] <0.228.0> Running boot step rabbit_auth_mechanism_cr_demo defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900016+00:00 [info] <0.228.0> Running boot step rabbit_queue_location_random defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900055+00:00 [info] <0.228.0> Running boot step rabbit_event defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900140+00:00 [info] <0.228.0> Running boot step rabbit_auth_mechanism_amqplain defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900186+00:00 [info] <0.228.0> Running boot step rabbit_auth_mechanism_plain defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900216+00:00 [info] <0.228.0> Running boot step rabbit_exchange_type_direct defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900249+00:00 [info] <0.228.0> Running boot step rabbit_exchange_type_fanout defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900280+00:00 [info] <0.228.0> Running boot step rabbit_exchange_type_headers defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900314+00:00 [info] <0.228.0> Running boot step rabbit_exchange_type_topic defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900345+00:00 [info] <0.228.0> Running boot step rabbit_mirror_queue_mode_all defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900373+00:00 [info] <0.228.0> Running boot step rabbit_mirror_queue_mode_exactly defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900407+00:00 [info] <0.228.0> Running boot step rabbit_mirror_queue_mode_nodes defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900440+00:00 [info] <0.228.0> Running boot step rabbit_priority_queue defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900464+00:00 [info] <0.228.0> Priority queues enabled, real BQ is rabbit_variable_queue[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900505+00:00 [info] <0.228.0> Running boot step rabbit_queue_location_client_local defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900550+00:00 [info] <0.228.0> Running boot step rabbit_queue_location_min_masters defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900590+00:00 [info] <0.228.0> Running boot step kernel_ready defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.900619+00:00 [info] <0.228.0> Running boot step rabbit_sysmon_minder defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.901120+00:00 [info] <0.228.0> Running boot step rabbit_epmd_monitor defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.901999+00:00 [info] <0.483.0> epmd monitor knows us, inter-node communication (distribution) port: 25672[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.902099+00:00 [info] <0.228.0> Running boot step guid_generator defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.903198+00:00 [info] <0.228.0> Running boot step rabbit_node_monitor defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.903325+00:00 [info] <0.487.0> Starting rabbit_node_monitor[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.903445+00:00 [info] <0.228.0> Running boot step delegate_sup defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.904240+00:00 [info] <0.228.0> Running boot step rabbit_memory_monitor defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.904411+00:00 [info] <0.228.0> Running boot step core_initialized defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.904447+00:00 [info] <0.228.0> Running boot step upgrade_queues defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.910771+00:00 [info] <0.228.0> Running boot step channel_tracking defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.910953+00:00 [info] <0.228.0> Setting up a table for channel tracking on this node: tracked_channel_on_node_rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.911112+00:00 [info] <0.228.0> Setting up a table for channel tracking on this node: tracked_channel_table_per_user_on_node_rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.911374+00:00 [info] <0.228.0> Running boot step rabbit_channel_tracking_handler defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.911414+00:00 [info] <0.228.0> Running boot step connection_tracking defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.911593+00:00 [info] <0.228.0> Setting up a table for connection tracking on this node: tracked_connection_on_node_rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.911717+00:00 [info] <0.228.0> Setting up a table for per-vhost connection counting on this node: tracked_connection_per_vhost_on_node_rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.911840+00:00 [info] <0.228.0> Setting up a table for per-user connection counting on this node: tracked_connection_table_per_user_on_node_rabbit@9698263dcd24[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.912008+00:00 [info] <0.228.0> Running boot step rabbit_connection_tracking_handler defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.912057+00:00 [info] <0.228.0> Running boot step rabbit_exchange_parameters defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.912116+00:00 [info] <0.228.0> Running boot step rabbit_mirror_queue_misc defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.912421+00:00 [info] <0.228.0> Running boot step rabbit_policies defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.913284+00:00 [info] <0.228.0> Running boot step rabbit_policy defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.913479+00:00 [info] <0.228.0> Running boot step rabbit_queue_location_validator defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.913630+00:00 [info] <0.228.0> Running boot step rabbit_quorum_memory_manager defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.913754+00:00 [info] <0.228.0> Running boot step rabbit_stream_coordinator defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.913959+00:00 [info] <0.228.0> Running boot step rabbit_vhost_limit defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.914030+00:00 [info] <0.228.0> Running boot step rabbit_mgmt_reset_handler defined by app rabbitmq_management[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.914078+00:00 [info] <0.228.0> Running boot step rabbit_mgmt_db_handler defined by app rabbitmq_management_agent[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.914103+00:00 [info] <0.228.0> Management plugin: using rates mode 'basic'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.914380+00:00 [info] <0.228.0> Running boot step recovery defined by app rabbit[0m
[36;1mrabbitmq       |[0m [38;5;160m2022-04-17 01:30:48.914354+00:00 [error] <0.228.0> Discarding message {'$gen_cast',{force_event_refresh,#Ref<0.1966004920.2827223041.82067>}} from <0.228.0> to <0.540.0> in an old incarnation (1650158898) of this node (1650159046)[0m
[36;1mrabbitmq       |[0m [38;5;160m2022-04-17 01:30:48.914354+00:00 [error] <0.228.0> [0m
[36;1mrabbitmq       |[0m [38;5;160m2022-04-17 01:30:48.914354+00:00 [error] <0.228.0> [0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.915454+00:00 [info] <0.525.0> Making sure data directory '/var/lib/rabbitmq/mnesia/rabbit@9698263dcd24/msg_stores/vhosts/628WB79CIFDYO9LJI6DKMI09L' for vhost '/' exists[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.916927+00:00 [info] <0.525.0> Starting message stores for vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.917037+00:00 [info] <0.529.0> Message store "628WB79CIFDYO9LJI6DKMI09L/msg_store_transient": using rabbit_msg_store_ets_index to provide index[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.919390+00:00 [info] <0.525.0> Started message store of type transient for vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.919584+00:00 [info] <0.533.0> Message store "628WB79CIFDYO9LJI6DKMI09L/msg_store_persistent": using rabbit_msg_store_ets_index to provide index[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.921830+00:00 [info] <0.525.0> Started message store of type persistent for vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.923975+00:00 [info] <0.525.0> Recovering 1 queues of type rabbit_classic_queue took 8ms[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.924020+00:00 [info] <0.525.0> Recovering 0 queues of type rabbit_quorum_queue took 0ms[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.924066+00:00 [info] <0.525.0> Recovering 0 queues of type rabbit_stream_queue took 0ms[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.925923+00:00 [info] <0.228.0> Running boot step empty_db_check defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.925962+00:00 [info] <0.228.0> Will not seed default virtual host and user: have definitions to load...[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.926005+00:00 [info] <0.228.0> Running boot step rabbit_looking_glass defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.926035+00:00 [info] <0.228.0> Running boot step rabbit_core_metrics_gc defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.926387+00:00 [info] <0.228.0> Running boot step background_gc defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.926742+00:00 [info] <0.228.0> Running boot step routing_ready defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.926826+00:00 [info] <0.228.0> Running boot step pre_flight defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.926989+00:00 [info] <0.228.0> Running boot step notify_cluster defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.927076+00:00 [info] <0.228.0> Running boot step networking defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.927179+00:00 [info] <0.228.0> Running boot step definition_import_worker_pool defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.927327+00:00 [info] <0.353.0> Starting worker pool 'definition_import_pool' with 96 processes in it[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.932192+00:00 [info] <0.228.0> Running boot step cluster_name defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.932241+00:00 [info] <0.228.0> Running boot step direct_client defined by app rabbit[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.932428+00:00 [info] <0.228.0> Running boot step rabbit_management_load_definitions defined by app rabbitmq_management[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.932668+00:00 [info] <0.660.0> Resetting node maintenance status[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.951700+00:00 [info] <0.719.0> Management plugin: HTTP (non-TLS) listener started on port 15672[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.951807+00:00 [info] <0.747.0> Statistics database started.[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.951886+00:00 [info] <0.746.0> Starting worker pool 'management_worker_pool' with 3 processes in it[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.960260+00:00 [info] <0.761.0> Prometheus metrics: HTTP (non-TLS) listener started on port 15692[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.960391+00:00 [info] <0.660.0> Ready to start client connection listeners[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:48.961667+00:00 [info] <0.805.0> started TCP listener on [::]:5672[0m
[33minference_fast_online_1 exited with code 1
[0m[36;1mrabbitmq       |[0m  completed with 4 plugins.
[36;1mrabbitmq       |[0m 2022-04-17 01:30:49.040468+00:00 [info] <0.660.0> Server startup complete; 4 plugins started.[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:49.040468+00:00 [info] <0.660.0>  * rabbitmq_prometheus[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:49.040468+00:00 [info] <0.660.0>  * rabbitmq_management[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:49.040468+00:00 [info] <0.660.0>  * rabbitmq_web_dispatch[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:49.040468+00:00 [info] <0.660.0>  * rabbitmq_management_agent[0m
[32;1mfast_online_1  |[0m None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
[36;1mrabbitmq       |[0m 2022-04-17 01:30:50.977324+00:00 [info] <0.810.0> accepting AMQP connection <0.810.0> (172.27.0.1:60606 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:50.980716+00:00 [info] <0.810.0> connection <0.810.0> (172.27.0.1:60606 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:50.987882+00:00 [info] <0.828.0> accepting AMQP connection <0.828.0> (172.27.0.1:60612 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:50.991020+00:00 [info] <0.828.0> connection <0.828.0> (172.27.0.1:60612 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:53.504107+00:00 [info] <0.847.0> accepting AMQP connection <0.847.0> (172.27.0.1:60632 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:53.508249+00:00 [info] <0.847.0> connection <0.847.0> (172.27.0.1:60632 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:53.516853+00:00 [info] <0.865.0> accepting AMQP connection <0.865.0> (172.27.0.1:60638 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:53.520119+00:00 [info] <0.865.0> connection <0.865.0> (172.27.0.1:60638 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:56.397617+00:00 [info] <0.884.0> accepting AMQP connection <0.884.0> (172.27.0.1:60684 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:56.401272+00:00 [info] <0.884.0> connection <0.884.0> (172.27.0.1:60684 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:56.408578+00:00 [info] <0.902.0> accepting AMQP connection <0.902.0> (172.27.0.1:60690 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:56.411023+00:00 [info] <0.902.0> connection <0.902.0> (172.27.0.1:60690 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:58.869193+00:00 [info] <0.921.0> accepting AMQP connection <0.921.0> (172.27.0.1:60710 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:58.872469+00:00 [info] <0.921.0> connection <0.921.0> (172.27.0.1:60710 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:58.880020+00:00 [info] <0.939.0> accepting AMQP connection <0.939.0> (172.27.0.1:60716 -> 172.27.0.5:5672)[0m
[36;1mrabbitmq       |[0m 2022-04-17 01:30:58.882853+00:00 [info] <0.939.0> connection <0.939.0> (172.27.0.1:60716 -> 172.27.0.5:5672): user 'guest' authenticated and granted access to vhost '/'[0m
[32;1mfast_online_1  |[0m INFO:     Started server process [1]
[32;1mfast_online_1  |[0m INFO:     Waiting for application startup.
[32;1mfast_online_1  |[0m INFO:     Application startup complete.
[32;1mfast_online_1  |[0m INFO:     Uvicorn running on http://0.0.0.0:8100 (Press CTRL+C to quit)
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:31 +0000] "GET / HTTP/1.1" 200 644 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:31 +0000] "GET /static/css/main.3eef771c.css HTTP/1.1" 304 0 "http://192.168.3.114:8080/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:31 +0000] "GET /static/js/main.66ae8e87.js HTTP/1.1" 304 0 "http://192.168.3.114:8080/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:31 +0000] "GET /static/css/main.3eef771c.css.map HTTP/1.1" 200 1517 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:31 +0000] "GET /base.drawio.png HTTP/1.1" 304 0 "http://192.168.3.114:8080/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:31 +0000] "GET /fine-tuning.png HTTP/1.1" 304 0 "http://192.168.3.114:8080/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:32 +0000] "GET /static/js/main.66ae8e87.js.map HTTP/1.1" 200 1258720 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:33 +0000] "OPTIONS /online/ner_rabbit HTTP/1.1" 502 559 "http://localhost:3000/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 2022/04/17 01:31:33 [error] 31#31: *11 connect() failed (111: Connection refused) while connecting to upstream, client: 10.32.12.10, server: 192.168.3.114, request: "OPTIONS /online/ner_rabbit HTTP/1.1", upstream: "http://172.27.0.3:8080//ner_rabbit", host: "192.168.3.114:8080", referrer: "http://localhost:3000/"
[32mnginx_proxy_1  |[0m 2022/04/17 01:31:36 [error] 31#31: *11 connect() failed (111: Connection refused) while connecting to upstream, client: 10.32.12.10, server: 192.168.3.114, request: "OPTIONS /online/ner_rabbit HTTP/1.1", upstream: "http://172.27.0.3:8080//ner_rabbit", host: "192.168.3.114:8080", referrer: "http://localhost:3000/"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:36 +0000] "OPTIONS /online/ner_rabbit HTTP/1.1" 502 559 "http://localhost:3000/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 2022/04/17 01:31:36 [error] 31#31: *11 connect() failed (111: Connection refused) while connecting to upstream, client: 10.32.12.10, server: 192.168.3.114, request: "OPTIONS /online/ner_rabbit HTTP/1.1", upstream: "http://172.27.0.3:8080//ner_rabbit", host: "192.168.3.114:8080", referrer: "http://localhost:3000/"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:36 +0000] "OPTIONS /online/ner_rabbit HTTP/1.1" 502 559 "http://localhost:3000/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:01:31:37 +0000] "OPTIONS /online/ner_rabbit HTTP/1.1" 502 559 "http://localhost:3000/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 2022/04/17 01:31:37 [error] 31#31: *11 connect() failed (111: Connection refused) while connecting to upstream, client: 10.32.12.10, server: 192.168.3.114, request: "OPTIONS /online/ner_rabbit HTTP/1.1", upstream: "http://172.27.0.3:8080//ner_rabbit", host: "192.168.3.114:8080", referrer: "http://localhost:3000/"
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65038 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:31:41.769+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159101:769230][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 3, snapshot max: 3 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65111 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:32:41.772+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159161:772904][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 6, snapshot max: 6 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65188 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:33:41.776+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159221:776783][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 8, snapshot max: 8 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65273 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:34:41.779+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159281:779748][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 10, snapshot max: 10 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65302 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:35:41.782+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159341:782635][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 12, snapshot max: 12 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65334 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:36:41.785+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159401:785001][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 14, snapshot max: 14 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65367 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:37:41.787+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159461:787779][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 16, snapshot max: 16 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65397 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:38:41.790+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159521:790210][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 18, snapshot max: 18 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65428 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:39:41.792+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159581:792812][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 20, snapshot max: 20 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65459 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:40:41.795+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159641:795642][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 22, snapshot max: 22 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65492 - "OPTIONS /infer_results HTTP/1.1" 200 OK
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65492 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:41:41.798+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159701:798374][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 24, snapshot max: 24 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:65526 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:42:41.801+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159761:801135][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 26, snapshot max: 26 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49169 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:43:41.804+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159821:804104][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 28, snapshot max: 28 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49204 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:44:41.806+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159881:806797][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 30, snapshot max: 30 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49234 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:45:41.809+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650159941:809151][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 32, snapshot max: 32 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49267 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:46:41.811+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160001:811623][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 34, snapshot max: 34 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49301 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:47:41.814+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160061:814669][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 36, snapshot max: 36 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49331 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:48:41.817+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160121:817116][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 38, snapshot max: 38 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49365 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:49:41.819+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160181:819888][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 40, snapshot max: 40 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49396 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:50:41.822+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160241:822498][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 42, snapshot max: 42 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49691 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:51:41.824+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160301:824966][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 44, snapshot max: 44 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49724 - "OPTIONS /infer_results HTTP/1.1" 200 OK
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49724 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:52:41.827+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160361:827490][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 46, snapshot max: 46 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49750 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:53:41.830+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160421:830631][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 48, snapshot max: 48 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49780 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:54:41.833+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160481:832992][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 50, snapshot max: 50 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49814 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:55:41.835+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160541:835835][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 52, snapshot max: 52 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49844 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:56:41.838+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160601:838232][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 54, snapshot max: 54 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49874 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:57:41.840+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160661:840668][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 56, snapshot max: 56 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49907 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:58:41.843+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160721:843130][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 58, snapshot max: 58 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49936 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T01:59:41.846+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160781:846549][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 60, snapshot max: 60 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49965 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:00:41.849+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160841:849448][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 62, snapshot max: 62 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:49996 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:01:41.852+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160901:852113][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 64, snapshot max: 64 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50029 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:02:41.854+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650160961:854728][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 66, snapshot max: 66 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50056 - "OPTIONS /infer_results HTTP/1.1" 200 OK
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50056 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:03:41.857+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161021:857241][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 68, snapshot max: 68 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50084 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:04:41.860+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161081:860043][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 70, snapshot max: 70 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50130 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:05:41.862+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161141:862430][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 72, snapshot max: 72 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50160 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:06:41.865+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161201:865112][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 74, snapshot max: 74 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[32mnginx_proxy_1  |[0m 2022/04/17 02:07:30 [error] 31#31: *16 connect() failed (111: Connection refused) while connecting to upstream, client: 10.32.12.10, server: 192.168.3.114, request: "OPTIONS /online/ner_rabbit HTTP/1.1", upstream: "http://172.27.0.3:8080//ner_rabbit", host: "192.168.3.114:8080", referrer: "http://localhost:3000/"
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:02:07:30 +0000] "OPTIONS /online/ner_rabbit HTTP/1.1" 502 559 "http://localhost:3000/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50202 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:07:41.867+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161261:867700][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 76, snapshot max: 76 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50248 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:08:41.870+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161321:870014][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 78, snapshot max: 78 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
[32mnginx_proxy_1  |[0m 10.32.12.10 - - [17/Apr/2022:02:08:49 +0000] "OPTIONS /online/bart_rabbit HTTP/1.1" 502 559 "http://localhost:3000/" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
[32mnginx_proxy_1  |[0m 2022/04/17 02:08:49 [error] 31#31: *18 connect() failed (111: Connection refused) while connecting to upstream, client: 10.32.12.10, server: 192.168.3.114, request: "OPTIONS /online/bart_rabbit HTTP/1.1", upstream: "http://172.27.0.3:8080//bart_rabbit", host: "192.168.3.114:8080", referrer: "http://localhost:3000/"
[36mfast_batch_1   |[0m INFO:     10.32.12.10:50300 - "GET /infer_results HTTP/1.1" 404 Not Found
[34mmongodb        |[0m {"t":{"$date":"2022-04-17T02:09:41.872+00:00"},"s":"I",  "c":"STORAGE",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":"[1650161381:872579][1:0x7f7f5b471700], WT_SESSION.checkpoint: [WT_VERB_CHECKPOINT_PROGRESS] saving checkpoint snapshot min: 80, snapshot max: 80 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 3778"}}
Stopping inference_nginx_proxy_1   ... 
Stopping inference_fast_online_1   ... 
Stopping inference_fast_batch_1    ... 
Stopping inference_node_1          ... 
Stopping mongodb                   ... 
Stopping rabbitmq                  ... 
Aborting.
Gracefully stopping... (press Ctrl+C again to force)
