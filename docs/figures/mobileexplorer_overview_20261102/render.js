// Render overview.svg to overview.png (4x) and a vector overview.pdf sized to the figure.
// Usage: node render.js [file.svg]   (Chromium via the preinstalled Playwright)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const svgPath = path.resolve(process.argv[2] || path.join(__dirname, 'overview.svg'));
  const base = svgPath.replace(/\.svg$/, '');
  const svg = fs.readFileSync(svgPath, 'utf8');
  const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  const [w, h] = [parseFloat(m[1]), parseFloat(m[2])];
  const html = `<!doctype html><html><head><style>
    @page { size: ${w}px ${h}px; margin: 0 }
    html, body { margin: 0; padding: 0; background: white }
    svg { display: block }
  </style></head><body>${svg}</body></html>`;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: Math.ceil(w), height: Math.ceil(h) }, deviceScaleFactor: 4 });
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: `${base}.png`, clip: { x: 0, y: 0, width: w, height: h } });
  await page.pdf({ path: `${base}.pdf`, width: `${w}px`, height: `${h}px`, printBackground: true, pageRanges: '1' });
  // report text that runs outside its enclosing rect (cheap overflow check)
  const issues = await page.evaluate(() => {
    const out = [];
    const texts = [...document.querySelectorAll('text')];
    const boxes = [...document.querySelectorAll('rect')].map(r => r.getBBox()).filter(b => b.width < 900);
    for (const t of texts) {
      const b = t.getBBox();
      const host = boxes.filter(r => b.x >= r.x - 0.5 && b.y >= r.y - 0.5 && b.x <= r.x + r.width && b.y <= r.y + r.height)
        .sort((p, q) => p.width * p.height - q.width * q.height)[0];
      if (host && (b.x + b.width > host.x + host.width + 0.5 || b.y + b.height > host.y + host.height + 0.5)) {
        out.push(`${t.textContent.slice(0, 50)} | text ${b.x.toFixed(1)}..${(b.x + b.width).toFixed(1)} vs box ..${(host.x + host.width).toFixed(1)}`);
      }
    }
    return out;
  });
  if (issues.length) console.log('possible overflow:\n  ' + issues.join('\n  '));
  await browser.close();
  console.log(`wrote ${base}.png and ${base}.pdf (${w}x${h})`);
})();
