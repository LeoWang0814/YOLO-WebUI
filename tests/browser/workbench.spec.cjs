const { test, expect } = require('@playwright/test');
const { execFileSync } = require('node:child_process');
const path = require('node:path');

const pages = ['/', '/?operation=predict', '/runs', ...[
  'getting-started', 'overview', 'datasets', 'train', 'predict', 'models',
  'runs', 'configuration', 'runtime', 'storage-and-privacy', 'troubleshooting', 'glossary',
].map(slug => `/docs/${slug}`)];

async function language(page, lang) {
  await page.evaluate(lang => window.WorkbenchI18n.setLanguage(lang), lang);
}

test('every page has Chinese copy and returns to English without retaining Chinese', async ({ page }) => {
  test.setTimeout(120000);
  // Names, file paths, units, library names and format identifiers are data.
  const properNames = ['YOLO', 'YOLOv10 Workbench', 'v10', 'Ultralytics', 'Ultralytics CLI',
    'CPU', 'GPU', 'COCO / COCO-MMDetection', 'CreateML', 'Pascal VOC',
    'TensorFlow Object Detection CSV', 'RetinaNet Keras CSV', 'YOLO v3 Keras / YOLO v4 PyTorch',
    'JSON', 'JSONL', 'XML', 'CSV', 'TXT', 'En'];
  for (const url of pages) {
    await page.goto(url);
    await page.waitForFunction(() => window.WorkbenchI18n);
    await language(page, 'en');
    const untranslated = await page.evaluate(properNames => {
      window.WorkbenchI18n.setLanguage('zh');
      const missing = [];
      const walker = document.createTreeWalker(document, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.parentElement?.closest('script, style, code, pre, [data-i18n-skip], [data-plot]')) continue;
        const value = node.nodeValue.trim();
        if (!/[A-Za-z]/.test(value) || /[\u3400-\u9fff]/.test(value)) continue;
        if (properNames.includes(value) || /^(?:GPU \d+|Torch [\w.+-]+|yolov\w+|v\d+[\d.]*|[\d.]+ MB)$/.test(value)) continue;
        missing.push(value);
      }
      return [...new Set(missing)];
    }, properNames);
    expect(untranslated, url).toEqual([]);
    await language(page, 'en');
    const residual = await page.evaluate(() => {
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      const texts = [];
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.parentElement?.closest('script, style, code, pre, [data-language-toggle], [data-i18n-skip], [data-plot]')) continue;
        if (/[\u3400-\u9fff]/.test(node.nodeValue)) texts.push(node.nodeValue.trim());
      }
      return texts;
    });
    expect(residual, url).toEqual([]);
  }
});

test('dynamic upload errors, tooltips and search results survive both language directions', async ({ page }) => {
  await page.goto('/');
  await language(page, 'zh');
  await page.locator('[data-dataset-upload]').evaluate(zone => {
    const data = new DataTransfer();
    data.items.add(new File(['not a zip'], 'sample.rar'));
    zone.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer: data }));
  });
  await expect(page.locator('[data-dataset-upload-detail]')).toHaveText('这里只能拖入 .zip 数据集压缩包。');
  await language(page, 'en');
  await expect(page.locator('[data-dataset-upload-detail]')).toHaveText('Only .zip dataset archives can be dropped here.');
  await expect(page.locator('[data-theme-toggle]')).toHaveAttribute('title', /Switch to (dark|light) theme/);
  await expect(page.locator('#command-preview')).toContainText('Complete the required fields');
  await language(page, 'zh');
  await expect(page.locator('#command-preview')).toContainText('请完成必填项');
  await page.goto('/docs');
  await language(page, 'zh');
  await page.locator('[data-doc-search]').fill('model');
  await expect(page.locator('[data-doc-search-result]').first()).toBeVisible();
  await language(page, 'en');
  await expect(page.locator('[data-doc-search-results]')).not.toContainText(/[\u3400-\u9fff]/);
  await page.locator('[data-doc-search]').fill('no-match-123');
  await language(page, 'zh');
  await expect(page.locator('.docs-search-empty')).toHaveText('没有与“no-match-123”匹配的文档。');
  await language(page, 'en');
  await expect(page.locator('.docs-search-empty')).toHaveText('No documentation matches “no-match-123”.');
});

test('asynchronous fragments translate and updated source text is not reverted', async ({ page }) => {
  await page.goto('/');
  await language(page, 'zh');
  await page.evaluate(() => {
    const element = document.querySelector('#dataset-validation');
    element.innerHTML = '<span>Dataset folder was not found or is not a directory.</span>';
    document.body.dispatchEvent(new CustomEvent('htmx:oobAfterSwap'));
  });
  await expect(page.locator('#dataset-validation')).toContainText('未找到数据集文件夹');
  await language(page, 'en');
  await expect(page.locator('#dataset-validation')).toHaveText('Dataset folder was not found or is not a directory.');
  await page.evaluate(() => {
    const label = document.querySelector('#dataset-validation span');
    label.firstChild.nodeValue = 'Dataset ready';
    window.WorkbenchI18n.apply(label);
  });
  await language(page, 'zh');
  await expect(page.locator('#dataset-validation')).toHaveText('数据集已就绪');
});

test('live charts render backend YOLOv10 data from the first epoch and refresh through HTMX', async ({ page }) => {
  const snapshots = JSON.parse(execFileSync(process.env.WORKBENCH_PYTHON || 'python', ['-c', `
import json, tempfile
from pathlib import Path
import pandas as pd
from core.workflows import metrics_snapshot
out = []
with tempfile.TemporaryDirectory() as directory:
    for n in (1, 2):
        columns = {f'{split}/{loss}_{head}': [1.0 / (i + 1) for i in range(n)]
            for split in ('train', 'val') for loss in ('box', 'cls', 'dfl') for head in ('om', 'oo')}
        columns.update({'epoch': list(range(1, n + 1)), 'metrics/mAP50(B)': [0.5] * n,
            'metrics/mAP50-95(B)': [0.3] * n, 'metrics/precision(B)': [0.6] * n,
            'metrics/recall(B)': [0.7] * n, 'lr/pg0': [0.01] * n, 'lr/pg1': [0.01] * n, 'lr/pg2': [0.01] * n})
        pd.DataFrame(columns).to_csv(Path(directory) / 'results.csv', index=False)
        out.append(metrics_snapshot(Path(directory))['figures'])
print(json.dumps(out))
`], { cwd: path.resolve(__dirname, '../..'), encoding: 'utf8' }));
  let responses = 0;
  await page.route('**/fragments/jobs/browser-test/results', route => {
    const figures = snapshots[Math.min(responses++, 1)];
    const charts = Object.entries(figures).map(([key, data]) =>
      `<div class="chart ${key === 'loss' ? 'chart-wide' : ''}" data-chart="${key}" data-plot='${data.replaceAll("'", '&#39;')}'></div>`).join('');
    return route.fulfill({ contentType: 'text/html', body:
      `<section id="run-results" hx-get="/fragments/jobs/browser-test/results" hx-trigger="every 2s" hx-swap="outerHTML"><div class="chart-grid">${charts}</div></section>` });
  });
  await page.goto('/');
  await language(page, 'zh');
  await page.evaluate(() => window.htmx.ajax('GET', '/fragments/jobs/browser-test/results', { target: '#run-results', swap: 'outerHTML' }));
  await expect(page.locator('[data-chart="loss"] .scatterlayer .point')).not.toHaveCount(0);
  await expect.poll(() => page.locator('[data-chart="loss"]').evaluate(el => el.data?.[0]?.x?.length)).toBe(2);
  for (const [kind, count] of [['loss', 12], ['quality', 4], ['learning_rate', 3]]) {
    const chart = page.locator(`[data-chart="${kind}"]`);
    await expect.poll(() => chart.evaluate(el => el.data?.length)).toBe(count);
    await expect(chart.locator('.main-svg').first()).toBeVisible();
    const bounds = await chart.locator('.main-svg').first().boundingBox();
    expect(bounds.width).toBeGreaterThan(250);
    expect(bounds.height).toBeGreaterThan(200);
    await expect(chart.locator('.scatterlayer .js-line')).not.toHaveCount(0);
  }
  await language(page, 'en');
  await expect.poll(() => page.locator('[data-chart="loss"]').evaluate(el => el.data[0].name)).toBe('Train box loss (one-to-many)');
  await language(page, 'zh');
  await expect.poll(() => page.locator('[data-chart="loss"]').evaluate(el => el.data[0].name)).toContain('训练');
  await expect(page.locator('[data-chart="loss"] .legendtext').first()).toContainText('训练');
  for (const label of await page.locator('[data-chart="loss"] .legendtext').allTextContents()) {
    expect(label).toMatch(/[\u3400-\u9fff]/);
  }
  await page.locator('[data-theme-toggle]').click();
  await expect.poll(() => page.locator('[data-chart="loss"]').evaluate(el => el.layout.font.color)).toBe('#edf4f1');
  await page.locator('[data-chart="loss"]').screenshot({ path: test.info().outputPath('loss-chart.png') });
});
