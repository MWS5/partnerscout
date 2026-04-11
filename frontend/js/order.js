/**
 * PartnerScout AI — Trial Order Form
 * Handles form submission → API call → redirect to dashboard
 */

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://partnerscout-api-production.up.railway.app';

// ── Admin mode detection ─────────────────────────────────────────────────────
// Usage: jares-ai.com/partnerscout?admin=YOUR_SECRET
const _adminSecret = new URLSearchParams(window.location.search).get('admin') || '';
const IS_ADMIN = _adminSecret.length > 0;

if (IS_ADMIN) {
  console.log('[PartnerScout] Admin mode active — full results, no blur');
  document.title = '⚡ PartnerScout — Admin Mode';
}

// ── Toast notifications ──────────────────────────────────────────────────────

function showToast(message, type = 'success') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.remove(), 4000);
}

// ── Trial form submit ────────────────────────────────────────────────────────

const trialForm = document.getElementById('trialForm');
const trialBtn  = document.getElementById('trialSubmitBtn');
const trialBtnText = document.getElementById('trialBtnText');

if (trialForm) {
  trialForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = document.getElementById('trialEmail').value.trim();
    if (!email) { showToast('Please enter your email', 'error'); return; }

    // Collect checked niches
    const nicheBoxes = document.querySelectorAll('input[name="niches"]:checked');
    const niches = Array.from(nicheBoxes).map(cb => cb.value);
    if (niches.length === 0) { showToast('Please select at least one category', 'error'); return; }

    // Parse regions
    const regionRaw = document.getElementById('trialRegion').value.trim();
    const regions = regionRaw
      ? regionRaw.split(',').map(r => r.trim()).filter(Boolean)
      : ['Nice', 'Cannes', 'Monaco'];

    const segment = document.getElementById('trialSegment').value;

    // Loading state
    trialBtn.classList.add('btn--loading');
    trialBtnText.innerHTML = '<span class="spinner"></span> Starting AI search...';

    try {
      const endpoint = IS_ADMIN
        ? `${API_BASE}/api/v1/orders/admin`
        : `${API_BASE}/api/v1/orders/trial`;

      const headers = { 'Content-Type': 'application/json' };
      if (IS_ADMIN) headers['X-Admin-Secret'] = _adminSecret;

      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          email, niches, regions, segment,
          count_target: IS_ADMIN ? 50 : 10,
          is_trial: !IS_ADMIN,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      const orderId = data.order_id;

      if (!orderId) throw new Error('No order ID returned');

      // Save to localStorage for dashboard to pick up
      localStorage.setItem('ps_trial_order_id', orderId);
      localStorage.setItem('ps_trial_email', email);
      if (IS_ADMIN) localStorage.setItem('ps_admin_secret', _adminSecret);

      const msg = IS_ADMIN
        ? '⚡ Admin mode — 50 full leads incoming!'
        : '✅ AI is searching! Redirecting to your results...';
      showToast(msg, 'success');

      const dashUrl = IS_ADMIN
        ? `dashboard.html?order_id=${orderId}&admin=${encodeURIComponent(_adminSecret)}`
        : `dashboard.html?order_id=${orderId}`;

      setTimeout(() => {
        window.location.href = dashUrl;
      }, 1500);

    } catch (err) {
      console.error('[ORDER] Trial submit error:', err);
      showToast(`Error: ${err.message}`, 'error');
      trialBtn.classList.remove('btn--loading');
      trialBtnText.textContent = '🔍 Run free preview';
    }
  });
}
