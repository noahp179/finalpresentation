FROM public.ecr.aws/lambda/python:3.10

# Install dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy your function code
COPY app ${LAMBDA_TASK_ROOT}


# Set the entrypoint
CMD ["lambda_function.lambda_handler"]