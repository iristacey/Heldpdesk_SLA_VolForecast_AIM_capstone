# Reproducible environment for the Help Desk Ticket SLA and Volume
# Forecasting capstone. This image is sufficient to run the project
# contract test suite (tests/test_project_contracts.py) out of the box.
#
# To also (re)generate the full pipeline, mount the raw dataset and the
# output folder as volumes at run time (see README.md "Container /
# reproducible environment"). The raw dataset is never baked into the
# image; retain original CC BY 4.0 and review privacy (see data/README.md).
FROM python:3.12-slim AS analysis

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the canonical output root exists even without a mounted volume.
RUN mkdir -p output data/raw

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

# Default to a lightweight, pinned test environment. Build the full analysis
# environment explicitly with `docker build --target analysis`.
FROM python:3.12-slim AS test

WORKDIR /app

COPY requirements-test.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-test.txt

COPY . .

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]