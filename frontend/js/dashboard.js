/**
 * PartnerScout AI — Dashboard
 * Polls order status → shows progress → renders results table
 */

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://partnerscout-api-production.up.railway.app';

const POLL_INTERVAL_MS = 3000;
const MAX_POLLS = 120; // 6 minutes max

// ── State ─────────────────────────────────────────────────────────────────────
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

    const emailCell = c.email === 'Not found'
      ? `<span class="locked">Not found</span>`
      : `<span class="email-blurred">${c.email}</span>`;

    const contactCell = (c.personal_email && c.personal_email.includes('🔒'))
      ? `<span class="locked"><span class="lock-icon">🔒</span> <a href="index.html#pricing">Unlock</a></span>`
      : `<span class="contact-blurred">${c.contact_person || 'Not found'}</span>`;

    return `
      <div class="result-row">
        <div>
          <div class="company-name">${c.company_name}</div>
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
async function showDone(orderId, isTrial) {
  // Update status card
  statusIcon.textContent = '✅';
  statusTitle.textContent = IS_ADMIN
    ? '⚡ Admin: full results ready!'
    : isTrial ? '10 preview leads ready!' : 'Your leads are ready!';
  statusSub.textContent = IS_ADMIN
    ? 'Full unblurred data — 50 companies'
    : isTrial ? 'Partial contacts shown — upgrade to unlock full data' : 'Download your full database below';
  statusBadge.textContent = 'Done';
  statusBadge.classList.add('status-badge--done');
  progressFill.style.width = '100%';
  updateProgressSteps(100);

  // Fetch preview results
  try {
    // Admin gets full JSON; trial gets blurred preview; paid gets full JSON
    const endpoint = IS_ADMIN || !isTrial
      ? `${API_BASE}/api/v1/export/${orderId}/json`
      : `${API_BASE}/api/v1/export/${orderId}/preview`;

    const fetchHeaders = IS_ADMIN ? { 'X-Admin-Secret': _adminSecret } : {};
    const resp = await fetch(endpoint, { headers: fetchHeaders });
    if (!resp.ok) throw new Error(`Failed to fetch results: ${resp.status}`);

    const data = await resp.json();
    const companies = Array.isArray(data) ? data : (data.companies || []);

    renderResults(companies, isTrial);

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    console.error('[DASHBOARD] Failed to load results:', err);
    // Still show results section but with empty table message
    resultsTable.innerHTML = `<div style="padding:24px;text-align:center;color:var(--text-muted)">Results are ready — check your email for the download link.</div>`;
    resultsSection.style.display = 'block';
  }
}

// ── Show error state ───────────────────────────────────────────────────────────
function showError(message) {
  statusIcon.textContent = '❌';
  statusTitle.textContent = 'Search failed';
  statusSub.textContent = 'An error occurred during the search';
  statusBadge.textContent = 'Error';
  statusBadge.classList.add('status-badge--error');
  progressWrap.style.display = 'none';
  if (errorMsg) errorMsg.textContent = message || 'Unknown error occurred.';
  if (errorSection) errorSection.style.display = 'block';
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
      progressFill.style.width = `${progress}%`;
      updateProgressSteps(progress);
    }

    if (status === 'done') {
      clearInterval(pollTimer);
      await showDone(orderId, is_trial !== false);
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
