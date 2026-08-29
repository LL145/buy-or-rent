/* 从 web/calculator.html 里抽出算法, 在 node 里对给定场景求值, 输出 JSON。
   由 tests/test_web_calculator.py 调用, 用来证明网页与 Python 同源。
   场景从 stdin 读入。 */
import fs from 'fs';
import path from 'path';

const root = path.resolve(import.meta.dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/calculator.html'), 'utf8');
const data = html.match(/<script id="windowData" type="application\/json">([\s\S]*?)<\/script>/)[1];
let js = html.match(/<script>\n([\s\S]*?)<\/script>\s*$/)[1];
js = js.replace('const DATA = JSON.parse(document.getElementById("windowData").textContent);',
                'const DATA = JSON.parse(process.env.WD);');
js = js.split('/* ---------- 视图 ---------- */')[0];
process.env.WD = data;
const m = await import('data:text/javascript,' + encodeURIComponent(
  js + '\nexport {simulate, breakevenGrowth, probExceed, evaluate, nominal};'));

const scenarios = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = scenarios.map(c => {
  const infl = c.infl_fixed;
  const s = {
    rentYield: c.rent_yield, down: c.down, mortRate: c.mort_rate,
    mortYears: c.mort_years, hold: c.hold, buyCost: c.buy_cost,
    sellCost: c.sell_cost, carry: c.carry, rInvestReal: c.r_invest_real,
    inflFixed: infl, infl, gRent: infl, rInvest: 0.06, gHouse: 0.02,
  };
  const gNom = m.breakevenGrowth({ ...s, rInvest: m.nominal(c.r_invest_real, infl) });
  const gReal = (1 + gNom) / (1 + infl) - 1;
  const hist = m.probExceed(gReal, c.hold, c.tercile);
  const ev = m.evaluate(s, c.tercile);
  return {
    g_star_real: gReal, p_hist: hist.p, n_hist: hist.n,
    p_buy_wins: ev.p, n_windows: ev.n,
    gap_p5: ev.p5, gap_median: ev.median, gap_p95: ev.p95,
    p_buy_ci: ev.ci,
  };
});
console.log(JSON.stringify(out));
