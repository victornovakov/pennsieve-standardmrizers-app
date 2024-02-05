# python-application-template

Ensure that your main executable file, e.g. `main.py` makes use of the `INPUT_DIR` and the `OUTPUT_DIR` ENVIRONMENT variables, to access input and to write output.

Add additional dependencies to the `Dockerfile`.

To run locally:

Run: `docker-compose up --build`

The above will create a `data` directory in your root directory locally.

The example copies files from the `INPUT_DIR` directory to the `OUTPUT_DIR` directory. The directories are set in `dev.env` and are defaulted to `/service/data/input` and `/service/data/ouput` for the input and output directories respectively.
To test, create `input` and `output` subfolders in the `data` directory. Create a test file (for example `test.txt`) in the `/service/data/input` directory. 

Re-Run: `docker-compose up --build`

The testfile should be copied to the `data/output` directory.

The current example supports python version 3.9.

