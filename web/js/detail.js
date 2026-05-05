function getId(){ return location.pathname.split('/').filter(Boolean).pop(); }
async function loadDetail(){
  const id=getId(); setText('eventId', id); const r=await API.get(`/api/logs/${id}`); const item=r.data;
  setImg('detailImage', item.annotated_image||item.original_image); document.getElementById('downloadBtn').href=item.annotated_image||item.original_image||'#'; document.getElementById('originalLink').href=item.original_image||'#'; document.getElementById('debugLink').href=item.debug_mask||'#';
  setText('detCount', pad2(item.total||0)); const conf=avgConfidence(item); setText('avgConf', confidenceText(conf)); setText('timestamp', fmtDate(item.timestamp_ms,item.datatime)); setText('storage', item.cloudinary_saved ? 'Cloudinary permanent storage' : 'Local temporary storage'); setText('breakTotal', `${item.total||0} total`);
  const rows=item.data||[]; document.getElementById('breakdown').innerHTML = rows.length ? rows.map(row=>`<div class="break-row"><div style="display:flex;align-items:center;gap:12px"><div class="iconbox" style="width:34px;height:34px"><span class="material-symbols-outlined" style="font-size:18px">bug_report</span></div><b>${escapeHtml(row.type)}</b></div><span class="det-count">${row.count}</span></div>`).join('') : '<div class="empty">No pest breakdown</div>';
}
loadDetail().catch(e=>alert(e.message));
