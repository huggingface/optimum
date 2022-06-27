# Helpful tips for testing & debugging optimum

## VSCODE

If you are using vscode you might have hare time discovering the test for the "testing" menu to run tests individually or debug them. You can copy the snippet below into `.vscode/settings.json`. 

```json
{
  "python.testing.pytestArgs": [
      "tests/onnxruntime",
      "tests/test_*"
  ],
  "python.testing.unittestEnabled": false,
  "python.testing.pytestEnabled": true
}
```

This snippet will discover all base tests and the tests inside the `tests/onnxruntime` folder.