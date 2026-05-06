let page = 1;
let pageSize = 10;
let last = null;

async function loadLogs(){
  const q = document.getElementById('search').value.trim();
  const p = document.getElementById('pestType').value.trim();

  const url = `/api/logs?page=${page}&page_size=${pageSize}${q ? `&search=${encodeURIComponent(q)}` : ''}${p ? `&pest_type=${encodeURIComponent(p)}` : ''}`;

  const r = await API.get(url);

  last = r;

  renderLogs(r.items || []);

  setText(
    'pageInfo',
    `Showing page ${r.page} of ${r.total_pages} · ${r.total_items} records`
  );

  document.getElementById('prevBtn').disabled = !r.has_prev;
  document.getElementById('nextBtn').disabled = !r.has_next;
}

function renderLogs(items){
  const tb = document.getElementById('logTable');

  if(!items.length){
    tb.innerHTML = '<tr><td colspan="6"><div class="empty">No logs found</div></td></tr>';
    return;
  }

  tb.innerHTML = items.map(item => {
    const type = primaryType(item);
    const typeTitle = primaryTypeTitle(item);
    const conf = avgConfidence(item);

    return `
      <tr class="log-row" onclick="location.href='/ui/logs/${item.id}'">
        <td>
          <img
            class="thumb sm"
            src="${item.annotated_image || item.original_image || ''}"
            onerror="imgFallback(event)"
          >
        </td>

        <td class="identification-cell">
          <b
            class="type-name"
            title="${escapeHtml(typeTitle)}"
            style="color:var(--primary)"
          >
            ${escapeHtml(type)}
          </b>
          <div class="sub">${escapeHtml(item.id || '')}</div>
        </td>

        <td>
          <div style="display:flex;align-items:center;gap:10px">
            <div class="confbar">
              <span style="width:${Math.round(conf * 100)}%"></span>
            </div>
            <b>${confidenceText(conf)}</b>
          </div>
        </td>

        <td>${item.total || 0}</td>

        <td>${fmtDate(item.timestamp_ms, item.datatime)}</td>

        <td style="text-align:right">
          <span class="material-symbols-outlined muted">open_in_new</span>
        </td>
      </tr>
    `;
  }).join('');
}

document.getElementById('filterBtn').onclick = () => {
  page = 1;
  loadLogs().catch(alert);
};

document.getElementById('search').addEventListener('keydown', e => {
  if(e.key === 'Enter'){
    page = 1;
    loadLogs().catch(alert);
  }
});

document.getElementById('prevBtn').onclick = () => {
  if(last?.has_prev){
    page--;
    loadLogs().catch(alert);
  }
};

document.getElementById('nextBtn').onclick = () => {
  if(last?.has_next){
    page++;
    loadLogs().catch(alert);
  }
};

loadLogs().catch(e => alert(e.message));