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

function fmtNum(n){
  return Number(n || 0).toLocaleString();
}

function pad2(n){
  return String(n || 0).padStart(2, '0');
}

function fmtDate(ms, fallback = ''){
  if(!ms) return fallback || '-';

  const d = new Date(Number(ms));

  return d.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function fmtTime(ms){
  if(!ms) return '-';

  return new Date(Number(ms)).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  });
}

function isUnknownType(type){
  return String(type || '').trim().toLowerCase() === 'unknown_pest';
}

function knownTypesFromData(item){
  const data = item?.data || [];

  return data
    .filter(row => !isUnknownType(row.type))
    .sort((a, b) => Number(b.count || 0) - Number(a.count || 0))
    .map(row => row.type)
    .filter(Boolean);
}

function primaryType(item){
  const data = item?.data || [];

  if(!data.length) return 'No pest';

  const sorted = [...data].sort((a, b) => Number(b.count || 0) - Number(a.count || 0));
  const top = sorted[0];

  if(!top) return 'No pest';

  const knownTypes = knownTypesFromData(item);

  // Main rule:
  // If unknown_pest is the highest count, show the known pest names instead.
  if(isUnknownType(top.type) && knownTypes.length){
    return knownTypes.join(', ');
  }

  return top.type || 'unknown_pest';
}

function primaryTypeTitle(item){
  const data = item?.data || [];

  if(!data.length) return 'No pest';

  const knownTypes = knownTypesFromData(item);
  const sorted = [...data].sort((a, b) => Number(b.count || 0) - Number(a.count || 0));
  const top = sorted[0];

  if(top && isUnknownType(top.type) && knownTypes.length){
    return knownTypes.join(', ');
  }

  return top?.type || 'unknown_pest';
}

function avgConfidence(item){
  if(item && item.avg_confidence !== undefined && item.avg_confidence !== null){
    return Number(item.avg_confidence || 0);
  }

  const dets = item?.detections || [];

  if(!dets.length) return 0;

  return dets.reduce((s, d) => s + Number(d.confidence || 0), 0) / dets.length;
}

function topConfidence(item){
  if(item && item.top_confidence !== undefined && item.top_confidence !== null){
    return Number(item.top_confidence || 0);
  }

  const dets = item?.detections || [];

  if(!dets.length) return 0;

  return Math.max(...dets.map(d => Number(d.confidence || 0)));
}

function confidenceText(v){
  return `${Math.round(Number(v || 0) * 100)}%`;
}

function imgFallback(e){
  e.target.src = 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" width="600" height="400">
      <rect width="100%" height="100%" fill="#e9efec"/>
      <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#52655d" font-family="Arial" font-size="22">No image</text>
    </svg>
  `);
}

function setText(id, value){
  const el = document.getElementById(id);
  if(el) el.textContent = value;
}

function setImg(id, src){
  const el = document.getElementById(id);

  if(el){
    el.src = src || '';
    el.onerror = imgFallback;
  }
}

function escapeHtml(s){
  return String(s ?? '').replace(/[&<>"]/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;'
  }[c]));
}