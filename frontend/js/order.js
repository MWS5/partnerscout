/**
 * PartnerScout AI — Trial Order Form
 * Handles form submission → API call → redirect to dashboard
 */

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://partnerscout-api-production.up.railway.app';

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
      const response = await fetch(`${API_BASE}/api/v1/orders/trial`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, niches, regions, segment, count_target: 10, is_trial: true }),
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

      showToast('✅ AI is searching! Redirecting to your results...', 'success');

      setTimeout(() => {
        window.location.href = `dashboard.html?order_id=${orderId}`;
      }, 1500);

    } catch (err) {
      console.error('[ORDER] Trial submit error:', err);
      showToast(`Error: ${err.message}`, 'error');
      trialBtn.classList.remove('btn--loading');
      trialBtnText.textContent = '🔍 Run free preview';
    }
  });
}
