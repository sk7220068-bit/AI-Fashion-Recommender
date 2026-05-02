import { test, expect } from '@playwright/test';

test('polling status transitions queued -> running -> completed', async ({ page }) => {
  let pollCount = 0;

  await page.route('**/api/upload-outfit', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ renderJobId: 'job_abc', renderStatus: 'queued' })
    });
  });

  await page.route('**/api/upgrade-jobs/job_abc**', async route => {
    pollCount += 1;
    const payload = pollCount < 2
      ? { jobId: 'job_abc', status: 'queued', progress: 0, stage: 'queued', result: { mainImageUrl: null, variants: [] } }
      : pollCount < 4
      ? { jobId: 'job_abc', status: 'running', progress: 65, stage: 'rendering', result: { mainImageUrl: null, variants: [] } }
      : { jobId: 'job_abc', status: 'completed', progress: 100, stage: 'done', result: { mainImageUrl: 'https://cdn/a.png', variants: ['https://cdn/b.png'] } };

    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) });
  });
  await page.goto('/');

  // Set file and occasion
  const fileChooserPromise = page.waitForEvent('filechooser');
  await page.locator('text=Click or drag an outfit photo here').click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles({
    name: 'test.jpg',
    mimeType: 'image/jpeg',
    buffer: Buffer.from('test')
  });

  await page.getByRole('button', { name: 'Casual' }).click();

  // Click analyse
  await page.locator('#btn-analyse').click();

  // Wait for the upload-outfit route to resolve
  await expect(page.locator('.loader-text')).toBeVisible();

  // It should show queued/pending initially
  await expect(page.locator('text=Rendering preview...').first()).toBeVisible({ timeout: 10000 });

  // Then it should eventually show the image
  await expect(page.locator('img[alt="Upgraded outfit preview"]')).toHaveAttribute('src', 'https://cdn/a.png', { timeout: 10000 });
  await expect(page.locator('img[alt="Upgrade variant 1"]')).toHaveAttribute('src', 'https://cdn/b.png');
});
