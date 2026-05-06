let dashboardCache = null;

async function loadDashboard(){
  const dash = await API.get('/api/dashboard');

  dashboardCache = dash;

  const s = dash.summary || {};
  const chart = dash.chart || {};

  setText('todayPests', fmtNum(s.today_pests));
  setText('biodiversity', fmtNum((s.top_pests || []).length));

  const avg = (s.last_7_days_pests || 0) / 7;
  setText('dailyAvg', Math.round(avg));

  const top = (s.top_pests || [])[0];
  setText('topToday', top ? `${fmtNum(top.count)} ${top.type} total` : 'No detections yet');
  setText('monthInfo', `${fmtNum(s.last_7_days_pests)} pests in last 7 days`);

  renderLive(dash.live_camera_stream?.latest);
  renderChart(chart.hourly_today || []);
  renderRecent(dash.logs || []);
}

async function loadLatest(){
  try{
    const r = await API.get('/api/live/latest');
    renderLive(r.latest);
  }catch(e){
    console.warn(e);
  }
}

function renderLive(item){
  const list = document.getElementById('liveList');

  if(!item){
    setImg('liveImage', '');
    list.innerHTML = '<div class="empty">No live detection yet</div>';
    return;
  }

  setImg('liveImage', item.annotated_image || item.original_image);

  document.getElementById('openLatest').href = `/ui/logs/${item.id}`;

  const data = item.data || [];
  const total = item.total || 0;

  setText('updateAge', fmtTime(item.timestamp_ms));

  list.innerHTML = data.length
    ? data.map((d, i) => `
      <div class="det-row">
        <div class="det-left">
          <span
            class="det-dot"
            style="background:${['#10b981', '#f59e0b', '#3b82f6', '#a855f7', '#ef4444'][i % 5]}"
          ></span>
          <span>${escapeHtml(d.type)}</span>
        </div>
        <span class="det-count">${pad2(d.count)}</span>
      </div>
    `).join('')
    : '<div class="empty">No pest in latest frame</div>';

  const pct = Math.min(100, Math.round((total / 25) * 100));

  setText('loadPct', pct + '%');
  document.getElementById('loadBar').style.width = pct + '%';
}

function renderChart(rows){
  const el = document.getElementById('hourlyChart');
  const max = Math.max(1, ...rows.map(r => r.total || 0));

  el.innerHTML = rows.map(r => `
    <div
      title="${r.hour}: ${r.total}"
      class="bar ${r.total === max ? 'hot' : ''}"
      style="height:${Math.max(4, (r.total / max) * 92)}%"
    ></div>
  `).join('');
}

function renderRecent(logs){
  const root = document.getElementById('recentLogs');

  setText('logCount', `Showing ${logs.length} latest entries`);

  if(!logs.length){
    root.innerHTML = '<div class="empty">No logs found</div>';
    return;
  }

  root.innerHTML = logs.map(item => {
    const type = primaryType(item);
    const typeTitle = primaryTypeTitle(item);
    const conf = avgConfidence(item);

    return `
      <div
        class="log-row"
        onclick="location.href='/ui/logs/${item.id}'"
        style="display:flex;align-items:center;padding:16px 20px;border-bottom:1px solid #f0f3f1"
      >
        <img
          class="thumb"
          src="${item.annotated_image || item.original_image || ''}"
          onerror="imgFallback(event)"
        >

        <div style="margin-left:16px;flex:1;min-width:0">
          <b
            class="type-name"
            title="${escapeHtml(typeTitle)}"
            style="color:var(--primary)"
          >
            ${escapeHtml(type)}
          </b>
          <div class="sub">Avg confidence ${confidenceText(conf)} · Total ${item.total || 0}</div>
        </div>

        <div class="desktop-only sub" style="min-width:180px">
          ${fmtDate(item.timestamp_ms, item.datatime)}
        </div>

        <span class="material-symbols-outlined muted">chevron_right</span>
      </div>
    `;
  }).join('');
}

document.getElementById('searchBox').addEventListener('input', async e => {
  const q = e.target.value.trim();

  if(!q){
    renderRecent(dashboardCache?.logs || []);
    return;
  }

  const r = await API.get(`/api/logs?page=1&page_size=10&search=${encodeURIComponent(q)}`);

  renderRecent(r.items || []);
});

document.getElementById('uploadBtn').addEventListener('click', async () => {
  const file = document.getElementById('uploadInput').files[0];

  if(!file){
    setText('uploadStatus', 'Choose image first.');
    return;
  }

  document.body.classList.add('loading');
  setText('uploadStatus', 'Analyzing image...');

  try{
    const r = await API.postImage('/api/analyze', file);
    setText('uploadStatus', `Saved. Total detected: ${r.total}. Opening detail...`);

    setTimeout(() => location.href = `/ui/logs/${r.id}`, 700);
  }catch(e){
    setText('uploadStatus', 'Failed: ' + e.message);
  }finally{
    document.body.classList.remove('loading');
  }
});

loadDashboard().catch(e => {
  console.error(e);
  document.body.insertAdjacentHTML(
    'afterbegin',
    `<div class="notice">Dashboard error: ${escapeHtml(e.message)}</div>`
  );
});

setInterval(loadLatest, 5000);