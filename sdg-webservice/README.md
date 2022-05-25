# SDG classifier web service

## Usage

To run (for development), edit `docker-compose.yml` to your liking, then:

````bash
sudo docker-compose up
````

By default the application runs on port 6007 and does not use a GPU.

To configure, set the following environment variables:

* `LISTEN_PORT`: the port at which the service will listen for requests
* `USE_GPU`: set to `YES` to attempt to use the GPU, any other value to disable GPU usage
* `MODEL_PATH`: path where the model files can be found (default: `../models`)
* `MODEL_FILE`: file that contains the model to be loaded (default: `model_2.bin`); this is relative to `MODEL_PATH`.
* `VOCAB_FILE`: file that contains the vocab to be loaded (default: `bert-base-uncased-vocab.txt`); this is relative to `MODEL_PATH`.
* `BATCH_SIZE`: the batch size to use for inferencing (default: 16)

The vocab file should obviously match the one used for the model at training time.
