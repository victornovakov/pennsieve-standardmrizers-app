# hackathon-py-example

Ensure that your python `main.py` takes an `input`directory and an `output` directory as arguments:

`main.py <INPUT_DIR> <OUTPUT_DIR>`

Add additional dependencies to the `Dockerfile`.

To run locally:

`docker-compose up --build`

The current example supports python version 3.9.

The example copies files from the `input` directory to the `output` directory. The `input` folder and `output` folders are created based on the `INTEGRATION_ID` specified in the `dev.env` file. On your first run of `docker-compose up --build` you will notice that a data folder is created with an `input/<INTEGRATION_ID>` folder, a similar output folder is also created. 
To test, copy a test file (for example `test.txt`) into the `input/1` directory and re-run the `docker-compose up --build` command. The testfile should be copied to the `output/1` directory.