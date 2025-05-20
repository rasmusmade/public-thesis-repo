import threading
pipeline_stop_event = threading.Event() # We use it to signal stopping across multiple threads and modules, so it's best to keep it here for modularity


