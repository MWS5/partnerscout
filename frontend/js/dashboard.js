/**
 * PartnerScout AI — Dashboard v3
 * - Trial: uses preview data from poll response directly (no extra fetch)
 * - Admin/paid: fetches full JSON from export endpoint
 * - Auto-reports JS errors to backend for JARVIS logging
 */

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://partnerscout-api-production.up.railway.app';

const POLL_INTERVAL_MS = 3000;
const MAX_POLLS = 250; // ~12.5 minutes max

let pollCount = 0;
let pollTimer = null;

const params       = new URLSearchParams(window.location.search);
const orderId      = params.get('order_id') || localStorage.getItem('ps_trial_order_id');
const _adminSecret = params.get('admin') || localStorage.getItem('ps_admin_secret') || '';
const IS_ADMIN     = _adminSecret.length > 0;

if (IS_ADMIN) {
  document.title = '⚡ PartnerScout — Admin Dashboard';
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusTitle    = document.getElementById('statusTitle');
const statusSub      = document.getElementById('statusSub');
const statusBadge    = document.getElementById('statusBadge');
const statusIcon     = document.getElementById('statusIcon');
const progressFill   = document.getElementById('progressFill');
const progressWrap   = document.getElementById('progressWrap');
const resultsSection = document.getElementById('resultsSection');
const errorSection   = document.getElementById('errorSection');
const errorMsg       = document.getElementById('errorMsg');
const resultsTable   = document.getElementById('resultsTable');
const resultsCount   = document.getElementById('resultsCount');
const trialBanner    = document.getElementById('trialBanner');

// ── Auto error reporting to JARVIS ────────────────────────────────────────────
async function reportError(context, message) {
  try {
    await fetch(`${API_BASE}/api/v1/log/error`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context, message, order_id: orderId, ts: new Date().toISOString() }),
    });
  } catch (_) { /* non-fatal */ }
}

// ── Progress step mapping ─────────────────────────────────────────────────────
const STEPS = [
  { id: 'step1', threshold: 10,  label: '🔎 Generating queries' },
  { id: 'step2', threshold: 30,  label: '📡 Searching sources' },
  { id: 'step3', threshold: 70,  label: '🏗️ Extracting data' },
  { id: 'step4', threshold: 85,  label: '✅ Validating luxury' },
  { id: 'step5', threshold: 100, label: '📦 Preparing results' },
];

function updateProgressSteps(progress) {
  STEPS.forEach((step) => {
    const el = document.getElementById(step.id);
    if (!el) return;
    if (progress >= step.threshold) {
      el.classList.remove('pstep--active');
      el.classList.add('pstep--done');
    } else if (progress >= (step.threshold - 20)) {
      el.classList.add('pstep--active');
    }
  });
}

// ── Category display ──────────────────────────────────────────────────────────
const CAT_LABELS = {
  hotel:        '🏨 Hotel',
  event_agency: '🎪 Event',
  wedding:      '💍 Wedding',
  concierge:    '🎩 Concierge',
  travel:       '✈️ Travel',
  venue:        '🏛️ Venue',
};

function catLabel(category) {
  return CAT_LABELS[category] || category;
}

// ── Luxury score badge ────────────────────────────────────────────────────────
function scoreBadge(score) {
  const pct = Math.round((score || 0) * 100);
  const cls = pct >= 80 ? 'score-badge--high' : 'score-badge--mid';
  const star = pct >= 90 ? '★★★' : pct >= 75 ? '★★' : '★';
  return `<span class="score-badge ${cls}">${star} ${pct}%</span>`;
}

// ── Render results table ──────────────────────────────────────────────────────
function renderResults(companies, isTrial) {
  if (!resultsTable) return;

  if (!companies || !companies.length) {
    resultsTable.innerHTML = `<div style="padding:32px;text-align:center;color:var(--text-muted)">
      No results found for this search. Try different niches or regions.
    </div>`;
    return;
  }

  const header = `
    <div class="result-row result-row--header">
      <span>Company</span>
      <span>Category</span>
      <span>Contact email</span>
      <span>Contact person</span>
      <span>Score</span>
    </div>
  `;

  const rows = companies.map((c) => {
    const website = c.website && c.website !== 'Not found'
      ? `<div class="company-website"><a href="${c.website}" target="_blank" rel="noopener">${c.website.replace(/^https?:\/\//, '').slice(0, 30)}…</a></div>`
      : '';
    const address = c.address && c.address !== 'Not found'
      ? `<div class="company-address">${c.address.slice(0, 50)}</div>`
      : '';

    const emailCell = (!c.email || c.email === 'Not found')
      ? `<span class="locked">Not found</span>`
      : `<span class="email-blurred">${c.email}</span>`;

    const contactCell = (c.personal_email && c.personal_email.includes('🔒'))
      ? `<span class="locked"><span class="lock-icon">🔒</span> <a href="index.html#pricing">Unlock</a></span>`
      : `<span class="contact-blurred">${c.contact_person || 'Not found'}</span>`;

    return `
      <div class="result-row">
        <div>
          <div class="company-name">${c.company_name || '—'}</div>
          ${website}
          ${address}
        </div>
        <div><span class="cat-tag">${catLabel(c.category)}</span></div>
        <div>${emailCell}</div>
        <div>${contactCell}</div>
        <div>${scoreBadge(c.luxury_score)}</div>
      </div>
    `;
  }).join('');

  resultsTable.innerHTML = header + rows;

  if (resultsCount) {
    resultsCount.textContent = `${companies.length} companies found`;
  }
}

// ── Show done state ────────────────────────────────────────────────────────────
/**
 * @param {string} orderId
 * @param {boolean} isTrial
 * @param {object} pollData - Full response from GET /orders/{id} (already has preview!)
 */
async function showDone(orderId, isTrial, pollData) {
  // Update status card
  if (statusIcon)  statusIcon.textContent = '✅';
  if (statusTitle) statusTitle.textContent = IS_ADMIN
    ? '⚡ Admin: full results ready!'
    : isTrial ? '10 preview leads ready!' : 'Your leads are ready!';
  if (statusSub) statusSub.textContent = IS_ADMIN
    ? 'Full unblurred data — 50 companies'
    : isTrial ? 'Partial contacts shown — upgrade to unlock full data' : 'Download your full database below';
  if (statusBadge) {
    statusBadge.textContent = 'Done';
    statusBadge.classList.add('status-badge--done');
  }
  if (progressFill) progressFill.style.width = '100%';
  updateProgressSteps(100);

  let companies = [];

  if (IS_ADMIN || !isTrial) {
    // ── Admin / paid: fetch full JSON export ──────────────────────────────
    try {
      const fetchHeaders = IS_ADMIN ? { 'X-Admin-Secret': _adminSecret } : {};
      const resp = await fetch(`${API_BASE}/api/v1/export/${orderId}/json`, { headers: fetchHeaders });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} from /json`);
      const data = await resp.json();
      companies = Array.isArray(data) ? data : (data.companies || []);
    } catch (err) {
      console.error('[DASHBOARD] Full export fetch failed:', err);
      await reportError('showDone/json', err.message);
    }

  } else {
    // ── Trial: use preview data already in the poll response ──────────────
    // GET /orders/{id} ALREADY returns response["preview"] for done+trial orders.
    // No second HTTP request needed — eliminates the extra failure point.
    companies = Array.isArray(pollData?.preview) ? pollData.preview : [];

    // Fallback: if poll data somehow missing preview, hit /preview endpoint
    if (!companies.length) {
      console.warn('[DASHBOARD] Poll data had no preview — trying /preview endpoint');
      try {
        const resp = await fetch(`${API_BASE}/api/v1/export/${orderId}/preview`);
        if (resp.ok) {
          const pData = await resp.json();
          companies = Array.isArray(pData.companies) ? pData.companies : [];
        } else {
          const errMsg = `HTTP ${resp.status} from /preview`;
          console.error('[DASHBOARD] Preview fallback failed:', errMsg);
          await reportError('showDone/preview_fallback', errMsg);
        }
      } catch (fbErr) {
        console.error('[DASHBOARD] Preview fallback exception:', fbErr);
        await reportError('showDone/preview_exception', fbErr.message);
      }
    }
  }

  // Render whatever we have
  renderResults(companies, isTrial);

  if (resultsSection) {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ── Show error state ───────────────────────────────────────────────────────────
function showError(message) {
  if (statusIcon)  statusIcon.textContent = '❌';
  if (statusTitle) statusTitle.textContent = 'Search failed';
  if (statusSub)   statusSub.textContent = 'An error occurred during the search';
  if (statusBadge) {
    statusBadge.textContent = 'Error';
    statusBadge.classList.add('status-badge--error');
  }
  if (progressWrap) progressWrap.style.display = 'none';
  if (errorMsg)     errorMsg.textContent = message || 'Unknown error occurred.';
  if (errorSection) errorSection.style.display = 'block';
  reportError('showError', message);
}

// ── Poll order status ─────────────────────────────────────────────────────────
async function pollStatus(orderId) {
  if (pollCount >= MAX_POLLS) {
    clearInterval(pollTimer);
    showError('Search timed out. Please try again or contact support.');
    return;
  }
  pollCount++;

  try {
    const resp = await fetch(`${API_BASE}/api/v1/orders/${orderId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const { status, progress, is_trial, error_msg } = data;

    // Update progress bar
    if (typeof progress === 'number') {
      if (progressFill) progressFill.style.width = `${progress}%`;
      updateProgressSteps(progress);
    }

    if (status === 'done') {
      clearInterval(pollTimer);
      // Pass full poll data — it contains data.preview for trial orders!
      await showDone(orderId, is_trial !== false, data);
    } else if (status === 'failed') {
      clearInterval(pollTimer);
      showError(error_msg || 'Pipeline failed. Please try again.');
    }
    // else: still running — keep polling

  } catch (err) {
    console.warn(`[DASHBOARD] Poll error (${pollCount}):`, err.message);
    // Don't stop polling on network hiccups — Railway cold starts can cause 502s
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
if (!orderId) {
  showError('No order ID found. Please start a new search.');
} else {
  // Start polling immediately, then every 3 seconds
  pollStatus(orderId);
  pollTimer = setInterval(() => pollStatus(orderId), POLL_INTERVAL_MS);
}
