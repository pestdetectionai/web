const API = {
  async get(path){
    const res = await fetch(path, {headers:{'Accept':'application/json'}});
    if(!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async postImage(path, file){
    const fd = new FormData();
    fd.append('image', file);
    const res = await fetch(path, {method:'POST', body:fd});
    if(!res.ok) throw new Error(await res.text());
    return res.json();
  }
};
function fmtNum(n){ return Number(n||0).toLocaleString(); }
function pad2(n){ return String(n||0).padStart(2,'0'); }
function fmtDate(ms, fallback=''){
  if(!ms) return fallback || '-';
  const d = new Date(Number(ms));
  return d.toLocaleString([], {year:'numeric',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'});
}
function fmtTime(ms){ if(!ms) return '-'; return new Date(Number(ms)).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
function primaryType(item){
  const data = item?.data || [];
  if(!data.length) return 'No pest';
  return [...data].sort((a,b)=>(b.count||0)-(a.count||0))[0].type || 'unknown_pest';
}
function avgConfidence(item){
  const dets = item?.detections || [];
  if(!dets.length) return 0;
  return dets.reduce((s,d)=>s + Number(d.confidence||0),0) / dets.length;
}
function confidenceText(v){ return `${Math.round(Number(v||0)*100)}%`; }
function imgFallback(e){
  e.target.src='data:image/svg+xml;charset=UTF-8,'+encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400"><rect width="100%" height="100%" fill="#e9efec"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#52655d" font-family="Arial" font-size="22">No image</text></svg>`);
}
function setText(id, value){ const el=document.getElementById(id); if(el) el.textContent=value; }
function setImg(id, src){ const el=document.getElementById(id); if(el){ el.src=src || ''; el.onerror=imgFallback; } }
function escapeHtml(s){ return String(s??'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
