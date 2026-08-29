"""Build a standalone HTML dashboard from scouting CSV outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOUTING_DIR = ROOT / "reports" / "scouting"
SHORTLIST_PATH = SCOUTING_DIR / "scouting_shortlist.csv"
CLUSTER_PATH = SCOUTING_DIR / "cluster_summary.csv"
DASHBOARD_PATH = SCOUTING_DIR / "dashboard.html"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _compact_shortlist(rows: list[dict]) -> list[dict]:
    keep = [
        "player_name",
        "role",
        "sample",
        "scouting_score",
        "impact_score",
        "impact_percentile",
        "profile_type",
        "closest_ipl_player",
        "closest_ipl_profile_type",
        "similarity_score",
        "strike_rate",
        "runs_per_innings",
        "boundary_pct",
        "economy",
        "bowling_sr",
        "wickets_per_innings",
        "scouting_reason",
    ]
    numeric = {
        "sample",
        "scouting_score",
        "impact_score",
        "impact_percentile",
        "similarity_score",
        "strike_rate",
        "runs_per_innings",
        "boundary_pct",
        "economy",
        "bowling_sr",
        "wickets_per_innings",
    }
    out = []
    for row in rows:
        item = {key: row.get(key, "") for key in keep}
        for key in numeric:
            item[key] = _to_float(item.get(key))
        out.append(item)
    return out


def _compact_clusters(rows: list[dict]) -> list[dict]:
    keep = [
        "role",
        "cluster",
        "profile_type",
        "players",
        "sma_players",
        "ipl_players",
        "mean_sample",
        "impact_score",
        "strike_rate",
        "runs_per_innings",
        "boundary_pct",
        "economy",
        "bowling_sr",
        "wickets_per_innings",
    ]
    numeric = set(keep) - {"role", "profile_type"}
    out = []
    for row in rows:
        item = {key: row.get(key, "") for key in keep}
        for key in numeric:
            item[key] = _to_float(item.get(key))
        out.append(item)
    return out


def build_dashboard() -> Path:
    shortlist = _compact_shortlist(_read_csv(SHORTLIST_PATH))
    clusters = _compact_clusters(_read_csv(CLUSTER_PATH))
    payload = json.dumps(
        {"shortlist": shortlist, "clusters": clusters},
        ensure_ascii=True,
        separators=(",", ":"),
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IPL Scouting Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #61717d;
      --line: #d8e0e5;
      --surface: #ffffff;
      --band: #f5f7f8;
      --green: #2f6f73;
      --copper: #b7653d;
      --blue: #315f9c;
      --gold: #b28b2e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--band);
      line-height: 1.45;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      position: sticky;
      top: 0;
      z-index: 4;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 20px 18px 36px;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 16px;
      padding: 16px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 10px;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 84px;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .kpi .value {{
      margin-top: 6px;
      font-size: 26px;
      font-weight: 700;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 160px 240px minmax(220px, 1fr);
      gap: 10px;
      align-items: end;
    }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
    }}
    select, input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 14px;
      background: white;
      color: var(--ink);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }}
    .chart-row {{
      display: grid;
      grid-template-columns: 210px 1fr 72px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }}
    .bar-bg {{
      height: 12px;
      background: #e9eef1;
      border-radius: 6px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      width: 0%;
      background: var(--green);
    }}
    .bar.bowl {{ background: var(--copper); }}
    .score {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .clusters {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .cluster {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .cluster strong {{
      display: block;
      margin-bottom: 6px;
    }}
    .cluster span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eef3f5;
      font-size: 12px;
      text-transform: uppercase;
      color: #384750;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .tag {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #f7fafb;
      white-space: nowrap;
    }}
    .reason {{
      min-width: 280px;
      color: #384750;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 4px 2px 0;
    }}
    @media (max-width: 900px) {{
      header {{ position: static; }}
      .kpis, .grid, .filters, .clusters {{
        grid-template-columns: 1fr;
      }}
      .chart-row {{
        grid-template-columns: 130px 1fr 56px;
      }}
      h1 {{ font-size: 21px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>IPL Scouting Dashboard</h1>
    <p class="subtitle">SMA domestic candidates ranked by observed impact, player profile, and IPL comparability.</p>
  </header>
  <main>
    <section>
      <div class="kpis" id="kpis"></div>
    </section>
    <section>
      <h2>Filters</h2>
      <div class="filters">
        <div>
          <label for="roleFilter">Role</label>
          <select id="roleFilter">
            <option value="all">All roles</option>
            <option value="bat">Batters</option>
            <option value="bowl">Bowlers</option>
          </select>
        </div>
        <div>
          <label for="profileFilter">Profile</label>
          <select id="profileFilter"></select>
        </div>
        <div>
          <label for="searchBox">Search</label>
          <input id="searchBox" type="search" placeholder="Player or IPL comparable">
        </div>
      </div>
    </section>
    <div class="grid">
      <section>
        <h2>Top Candidates</h2>
        <div id="topBars"></div>
      </section>
      <section>
        <h2>Player Profiles</h2>
        <div class="clusters" id="clusterCards"></div>
      </section>
    </div>
    <section>
      <h2>Scouting Shortlist</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Player</th>
              <th>Role</th>
              <th>Profile</th>
              <th>Score</th>
              <th>Impact %</th>
              <th>Sample</th>
              <th>Closest IPL Player</th>
              <th>Similarity</th>
              <th>Key Metrics</th>
              <th>Scouting Reason</th>
            </tr>
          </thead>
          <tbody id="shortlistBody"></tbody>
        </table>
      </div>
    </section>
    <footer>Generated from reports/scouting/scouting_shortlist.csv and cluster_summary.csv.</footer>
  </main>
  <script>
    const data = {payload};
    const shortlist = data.shortlist;
    const clusters = data.clusters;

    const roleFilter = document.getElementById('roleFilter');
    const profileFilter = document.getElementById('profileFilter');
    const searchBox = document.getElementById('searchBox');

    function fmt(value, digits = 1) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'NA';
      return Number(value).toFixed(digits);
    }}

    function roleLabel(role) {{
      return role === 'bat' ? 'Batter' : 'Bowler';
    }}

    function filteredRows() {{
      const role = roleFilter.value;
      const profile = profileFilter.value;
      const q = searchBox.value.trim().toLowerCase();
      return shortlist.filter(row => {{
        const roleOk = role === 'all' || row.role === role;
        const profileOk = profile === 'all' || row.profile_type === profile;
        const text = `${{row.player_name}} ${{row.closest_ipl_player}} ${{row.profile_type}}`.toLowerCase();
        return roleOk && profileOk && (!q || text.includes(q));
      }}).sort((a, b) => b.scouting_score - a.scouting_score);
    }}

    function setProfileOptions() {{
      const role = roleFilter.value;
      const profiles = [...new Set(shortlist
        .filter(row => role === 'all' || row.role === role)
        .map(row => row.profile_type)
        .filter(Boolean))]
        .sort();
      const current = profileFilter.value;
      profileFilter.innerHTML = '<option value="all">All profiles</option>' +
        profiles.map(p => `<option value="${{p}}">${{p}}</option>`).join('');
      if (profiles.includes(current)) profileFilter.value = current;
    }}

    function renderKpis(rows) {{
      const batters = rows.filter(row => row.role === 'bat').length;
      const bowlers = rows.filter(row => row.role === 'bowl').length;
      const avgScore = rows.length ? rows.reduce((sum, row) => sum + row.scouting_score, 0) / rows.length : 0;
      const top = rows[0]?.player_name || 'NA';
      document.getElementById('kpis').innerHTML = [
        ['Candidates', rows.length],
        ['Batters', batters],
        ['Bowlers', bowlers],
        ['Avg Score', fmt(avgScore, 1)],
      ].map(([label, value]) => `<div class="kpi"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`).join('');
    }}

    function renderBars(rows) {{
      const top = rows.slice(0, 12);
      const maxScore = Math.max(...top.map(row => row.scouting_score), 1);
      document.getElementById('topBars').innerHTML = top.map(row => {{
        const pct = Math.max(2, row.scouting_score / maxScore * 100);
        const cls = row.role === 'bowl' ? 'bar bowl' : 'bar';
        return `<div class="chart-row">
          <div>${{row.player_name}}</div>
          <div class="bar-bg"><div class="${{cls}}" style="width:${{pct}}%"></div></div>
          <div class="score">${{fmt(row.scouting_score, 1)}}</div>
        </div>`;
      }}).join('');
    }}

    function renderClusters() {{
      document.getElementById('clusterCards').innerHTML = clusters.map(cluster => {{
        const metric = cluster.role === 'bat'
          ? `SR ${{fmt(cluster.strike_rate)}}, RPI ${{fmt(cluster.runs_per_innings)}}, boundary% ${{fmt(cluster.boundary_pct)}}`
          : `Econ ${{fmt(cluster.economy, 2)}}, BSR ${{fmt(cluster.bowling_sr)}}, WPI ${{fmt(cluster.wickets_per_innings, 2)}}`;
        return `<div class="cluster">
          <strong>${{roleLabel(cluster.role)}}: ${{cluster.profile_type}}</strong>
          <span>${{cluster.players}} players | SMA ${{cluster.sma_players}} | IPL ${{cluster.ipl_players}}</span>
          <span>Mean impact ${{fmt(cluster.impact_score, 2)}} | ${{metric}}</span>
        </div>`;
      }}).join('');
    }}

    function keyMetrics(row) {{
      if (row.role === 'bat') {{
        return `SR ${{fmt(row.strike_rate)}}<br>RPI ${{fmt(row.runs_per_innings)}}<br>Boundary ${{fmt(row.boundary_pct)}}%`;
      }}
      return `Economy ${{fmt(row.economy, 2)}}<br>BSR ${{fmt(row.bowling_sr)}}<br>WPI ${{fmt(row.wickets_per_innings, 2)}}`;
    }}

    function renderTable(rows) {{
      document.getElementById('shortlistBody').innerHTML = rows.map(row => `<tr>
        <td><strong>${{row.player_name}}</strong></td>
        <td>${{roleLabel(row.role)}}</td>
        <td><span class="tag">${{row.profile_type}}</span></td>
        <td class="num">${{fmt(row.scouting_score, 1)}}</td>
        <td class="num">${{fmt(row.impact_percentile, 1)}}</td>
        <td class="num">${{fmt(row.sample, 0)}}</td>
        <td>${{row.closest_ipl_player || 'NA'}}</td>
        <td class="num">${{fmt(row.similarity_score, 1)}}</td>
        <td>${{keyMetrics(row)}}</td>
        <td class="reason">${{row.scouting_reason || ''}}</td>
      </tr>`).join('');
    }}

    function render() {{
      setProfileOptions();
      const rows = filteredRows();
      renderKpis(rows);
      renderBars(rows);
      renderClusters();
      renderTable(rows);
    }}

    roleFilter.addEventListener('change', render);
    profileFilter.addEventListener('change', render);
    searchBox.addEventListener('input', render);
    render();
  </script>
</body>
</html>
"""

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH


def main() -> None:
    path = build_dashboard()
    print(f"Saved dashboard to {path}")


if __name__ == "__main__":
    main()
