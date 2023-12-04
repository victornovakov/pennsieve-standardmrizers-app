# hackathon-py-example

Ensure that your python file takes an input directory and an output directory

Add additional dependencies to the `Dockerfile`.

To run locally:

`docker-compose up --build`

The current example supports python version 3.9.

The example copies a files from the input directory to the output directory. The `input` folder and `output` folders are created based on the `INTEGRATION_ID`. On your first run of `docker-compose up --build` you will notice that a data folder is created with an `input/<INTEGRATION_ID>` folder, a similar output folder is also created. 
To test copy a test file into the `input/1` directory and re-run the `docker-compose up --build` command.