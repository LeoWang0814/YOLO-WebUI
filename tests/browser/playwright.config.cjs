const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: '.',
  testMatch: '*.spec.cjs',
  workers: 1,
  use: { baseURL: process.env.WORKBENCH_TEST_URL || 'http://127.0.0.1:6006', headless: true },
});
