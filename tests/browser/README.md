# Browser regression checks

With the Python dependencies installed, start a disposable Workbench instance:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 6006
```

In another terminal, install and run the browser checks (Node.js 18+):

```bash
cd tests/browser
npm ci
npx playwright install --with-deps chromium
npm test
```

`WORKBENCH_TEST_URL` can select another local service port. These checks mock
uploads and training responses; they do not start training or modify datasets.
They exercise both languages, asynchronous fragments, language switching,
documentation, and Plotly rendering. Stop the disposable service afterwards.

Run the checks with the project's Python environment activated. If `python`
points to a different environment, set `WORKBENCH_PYTHON` to the full path of
the project's Python executable; the chart checks use it to generate metric
snapshots with the real backend.
